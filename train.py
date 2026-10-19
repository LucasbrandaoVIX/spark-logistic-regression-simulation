"""Simulação didática de Map/Reduce para regressão logística com processos."""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

FEATURES = ["trip_distance", "fare_amount", "passenger_count", "duration_minutes", "pickup_hour"]


def sigmoid(z: np.ndarray) -> np.ndarray:
    exp_negative_abs = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + exp_negative_abs), exp_negative_abs / (1 + exp_negative_abs))


def loss_and_gradient(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple[float, np.ndarray]:
    """Binary cross-entropy e gradiente médio; weights inclui o intercepto."""
    n, gradient_sum, loss_sum = _partition_metrics(X, y, weights)
    return loss_sum / n, gradient_sum / n


def _partition_metrics(X_part: np.ndarray, y_part: np.ndarray, weights: np.ndarray) -> tuple[int, np.ndarray, float]:
    """Worker: MAP calcula somas locais de gradiente e loss (não médias)."""
    logits = X_part @ weights[1:] + weights[0]
    probabilities = sigmoid(logits)
    errors = probabilities - y_part
    loss_sum = np.sum(np.logaddexp(0, logits) - y_part * logits)
    gradient_sum = np.concatenate(([errors.sum()], X_part.T @ errors))
    return len(y_part), gradient_sum, float(loss_sum)


def tip_labels(tips: np.ndarray, fares: np.ndarray) -> np.ndarray:
    """Valores em centavos (arredondamento para o par); 5 * gorjeta >= tarifa."""
    tips, fares = np.asarray(tips, dtype=float), np.asarray(fares, dtype=float)
    if tips.shape != fares.shape or tips.ndim != 1:
        raise ValueError("Gorjetas e tarifas devem ser vetores do mesmo tamanho.")
    if not (np.isfinite(tips).all() and np.isfinite(fares).all()) or np.any(tips < 0) or np.any(fares <= 0):
        raise ValueError("Use gorjetas finitas >= 0 e tarifas finitas > 0.")
    tips_cents, fares_cents = np.rint(tips * 100), np.rint(fares * 100)
    # Margem para converter e multiplicar em int64 sem overflow.
    max_cents = np.iinfo(np.int64).max // 10
    if np.any(tips_cents > max_cents) or np.any(fares_cents > max_cents) or np.any(fares_cents < 1):
        raise ValueError("Valores monetários fora da faixa de centavos suportada.")
    return (5 * tips_cents.astype(np.int64) >= fares_cents.astype(np.int64)).astype(float)


def split_partitions(X: np.ndarray, y: np.ndarray, workers: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if workers < 1:
        raise ValueError("O número de workers precisa ser >= 1.")
    indices = np.array_split(np.arange(len(y)), workers)
    return [(X[idx], y[idx]) for idx in indices if len(idx)]


def train_sequential(X: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float) -> tuple[np.ndarray, list[float], float]:
    weights = np.zeros(X.shape[1] + 1, dtype=np.float64)
    history: list[float] = []
    start = time.perf_counter()
    for epoch in range(epochs + 1):
        loss, gradient = loss_and_gradient(X, y, weights)
        history.append(loss)
        if epoch < epochs:
            weights -= learning_rate * gradient
    return weights, history, time.perf_counter() - start


def train_distributed(X: np.ndarray, y: np.ndarray, workers: int, epochs: int, learning_rate: float) -> tuple[np.ndarray, list[float], float]:
    """Driver coordena; os arrays são particionados uma vez; parâmetros são enviados em cada época."""
    parts = split_partitions(X, y, workers)
    weights = np.zeros(X.shape[1] + 1, dtype=np.float64)
    history: list[float] = []
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for epoch in range(epochs + 1):
            # BROADCAST simulado: pesos serializados para cada tarefa do worker.
            futures = [pool.submit(_partition_metrics, xp, yp, weights) for xp, yp in parts]
            local_results = [future.result() for future in futures]
            # REDUCE: agrega somas de gradiente e loss e normaliza pelo total de exemplos.
            total_n = sum(n for n, _, _ in local_results)
            gradient_sum = sum((gradient for _, gradient, _ in local_results), start=np.zeros_like(weights))
            loss_sum = sum(loss for _, _, loss in local_results)
            history.append(loss_sum / total_n)
            global_gradient = gradient_sum / total_n
            if epoch < epochs:
                weights -= learning_rate * global_gradient
    return weights, history, time.perf_counter() - start


def prepare_data(path: Path, sample_size: int, seed: int, test_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    columns = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance", "fare_amount", "passenger_count", "tip_amount", "payment_type"]
    frame = pd.read_parquet(path, columns=columns)
    frame = frame.loc[frame.payment_type == 1, columns].copy()
    frame["tpep_pickup_datetime"] = pd.to_datetime(frame.tpep_pickup_datetime, errors="coerce")
    frame["tpep_dropoff_datetime"] = pd.to_datetime(frame.tpep_dropoff_datetime, errors="coerce")
    frame["duration_minutes"] = (frame.tpep_dropoff_datetime - frame.tpep_pickup_datetime).dt.total_seconds() / 60
    frame["pickup_hour"] = frame.tpep_pickup_datetime.dt.hour
    # Faixas amplas removem erros evidentes sem engenharia de features.
    valid = (
        frame["trip_distance"].between(0.1, 100)
        & frame["fare_amount"].between(1, 500)
        & frame["passenger_count"].between(1, 6)
        & frame["duration_minutes"].between(1, 240)
        & frame["tip_amount"].ge(0)
    )
    frame = frame.loc[valid, FEATURES + ["tip_amount"]].dropna()
    if len(frame) < 2:
        raise ValueError("Menos de duas corridas válidas após o pré-processamento.")
    if sample_size > 0 and len(frame) > sample_size:
        frame = frame.sample(n=sample_size, random_state=seed)
    y = tip_labels(frame.tip_amount.to_numpy(), frame.fare_amount.to_numpy())
    X = frame[FEATURES].to_numpy(dtype=np.float64)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def evaluate(weights: np.ndarray, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score
    predictions = (sigmoid(np.column_stack((np.ones(len(X)), X)) @ weights) >= 0.5).astype(int)
    return {"accuracy": float(accuracy_score(y, predictions)), "f1": float(f1_score(y, predictions, zero_division=0))}


def save_charts(results: list[dict], output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [f"{r['workers']} worker(s)" for r in results]
    plt.figure(figsize=(8, 5))
    plt.bar(labels, [r["seconds"] for r in results], color="#3274a1")
    plt.ylabel("Tempo de treinamento (s)")
    plt.title("Tempo versus quantidade de processos")
    plt.tight_layout()
    plt.savefig(output_dir / "tempo_por_workers.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    for r in results:
        plt.plot(range(len(r["loss"])), r["loss"], marker="o", label=f"{r['workers']} worker(s)")
    plt.xlabel("Época")
    plt.ylabel("Binary cross-entropy")
    plt.title("Convergência: sequencial e distribuído")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_por_epoca.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/yellow_tripdata_2025-01.parquet"))
    parser.add_argument("--sample-size", type=int, default=300_000, help="Use 0 para todas as linhas válidas; valores usuais: 100000, 300000, 500000 ou 1000000.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--workers", type=int, nargs="+", default=[2, 4, 8], help="Workers a comparar (a execução sequencial é sempre incluída).")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    if not args.data.exists():
        raise SystemExit(f"Arquivo Parquet não encontrado: {args.data}\nBaixe yellow_tripdata_2025-01.parquet do portal TLC indicado no README.")
    if args.epochs < 1 or args.learning_rate <= 0 or any(w < 2 for w in args.workers):
        raise SystemExit("Use epochs >= 1, learning-rate > 0 e workers >= 2.")

    X_train, X_test, y_train, y_test = prepare_data(args.data, args.sample_size, args.seed, args.test_size)
    print(f"Treino: {len(y_train):,} | teste: {len(y_test):,} | features: {len(FEATURES)}")
    results = []
    configurations = [(1, "sequencial")] + [(n, "distribuído") for n in args.workers]
    reference = None
    for workers, mode in configurations:
        if workers == 1:
            weights, history, seconds = train_sequential(X_train, y_train, args.epochs, args.learning_rate)
        else:
            weights, history, seconds = train_distributed(X_train, y_train, workers, args.epochs, args.learning_rate)
        metrics = evaluate(weights, X_test, y_test)
        if reference is None:
            reference = weights
        else:
            np.testing.assert_allclose(weights, reference, rtol=1e-9, atol=1e-10)
            np.testing.assert_allclose(history, results[0]["loss"], rtol=1e-10, atol=1e-12)
        result = {"workers": workers, "mode": mode, "seconds": seconds, "speedup": None, "loss": history, **metrics}
        results.append(result)
        print(f"{mode:11s} {workers:>2} processo(s) | {seconds:8.2f}s | acc={metrics['accuracy']:.4f} | F1={metrics['f1']:.4f}")
    baseline = results[0]["seconds"]
    for result in results:
        result["speedup"] = baseline / result["seconds"] if result["seconds"] else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_charts(results, args.output_dir)
    (args.output_dir / "resultados.json").write_text(json.dumps({"sample_size_requested": args.sample_size, "epochs": args.epochs, "learning_rate": args.learning_rate, "seed": args.seed, "results": results}, indent=2), encoding="utf-8")
    print(f"Gráficos e resultados salvos em: {args.output_dir.resolve()}")


if __name__ == "__main__":
    # Necessário para o modo spawn do macOS/Windows.
    main()
