"""Simulação didática de Map/Reduce para regressão logística com processos."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
import platform
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

FEATURES = [
    "trip_distance",
    "fare_amount",
    "passenger_count",
    "duration_minutes",
    "pickup_hour",
]


def sigmoid(z: np.ndarray) -> np.ndarray:
    exp_negative_abs = np.exp(-np.abs(z))
    return np.where(
        z >= 0, 1 / (1 + exp_negative_abs), exp_negative_abs / (1 + exp_negative_abs)
    )


def loss_and_gradient(
    X: np.ndarray, y: np.ndarray, weights: np.ndarray
) -> tuple[float, np.ndarray]:
    """Binary cross-entropy e gradiente médio; weights inclui o intercepto."""
    n, gradient_sum, loss_sum = _partition_metrics(X, y, weights)
    return loss_sum / n, gradient_sum / n


def _partition_metrics(
    X_part: np.ndarray, y_part: np.ndarray, weights: np.ndarray
) -> tuple[int, np.ndarray, float]:
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
    if (
        not (np.isfinite(tips).all() and np.isfinite(fares).all())
        or np.any(tips < 0)
        or np.any(fares <= 0)
    ):
        raise ValueError("Use gorjetas finitas >= 0 e tarifas finitas > 0.")
    tips_cents, fares_cents = np.rint(tips * 100), np.rint(fares * 100)
    # Margem para converter e multiplicar em int64 sem overflow.
    max_cents = np.iinfo(np.int64).max // 10
    if (
        np.any(tips_cents > max_cents)
        or np.any(fares_cents > max_cents)
        or np.any(fares_cents < 1)
    ):
        raise ValueError("Valores monetários fora da faixa de centavos suportada.")
    return (5 * tips_cents.astype(np.int64) >= fares_cents.astype(np.int64)).astype(
        float
    )


def split_partitions(
    X: np.ndarray, y: np.ndarray, workers: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    if workers < 1:
        raise ValueError("O número de workers precisa ser >= 1.")
    if workers > len(y):
        raise ValueError("Use no máximo um worker por exemplo de treino.")
    # Fatias contíguas evitam as cópias por indexação avançada no driver.
    return list(zip(np.array_split(X, workers), np.array_split(y, workers)))


def validate_training(
    X: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float
) -> None:
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or not len(y):
        raise ValueError(
            "X e y precisam ter dimensões compatíveis e pelo menos uma linha."
        )
    if (
        not np.isfinite(X).all()
        or not np.isfinite(y).all()
        or not np.isin(y, [0, 1]).all()
    ):
        raise ValueError("Features devem ser finitas e rótulos devem ser 0 ou 1.")
    if epochs < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Use epochs >= 1 e learning-rate finito > 0.")


def train_sequential(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    learning_rate: float,
    *,
    timings: dict | None = None,
) -> tuple[np.ndarray, list[float], float]:
    validate_training(X, y, epochs, learning_rate)
    weights = np.zeros(X.shape[1] + 1, dtype=np.float64)
    history: list[float] = []
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        ready = time.perf_counter()
        for epoch in range(epochs + 1):
            loss, gradient = loss_and_gradient(X, y, weights)
            history.append(loss)
            if epoch < epochs:
                weights -= learning_rate * gradient
        finished = time.perf_counter()
    total = time.perf_counter() - start
    if timings is not None:
        timings.update(
            partition_seconds=0.0,
            startup_seconds=ready - start,
            iteration_seconds=finished - ready,
            shutdown_seconds=total - (finished - start),
            total_seconds=total,
        )
    return weights, history, total


def _worker(
    connection, partition_id: int, X_part: np.ndarray, y_part: np.ndarray
) -> None:
    """Uma partição residente; só pesos entram e pequenas contribuições saem."""
    try:
        with threadpool_limits(limits=1):
            connection.send(
                {"partition": partition_id, "pid": os.getpid(), "rows": len(y_part)}
            )
            while True:
                weights = connection.recv()
                if weights is None:
                    break
                connection.send(_partition_metrics(X_part, y_part, weights))
    except Exception as exc:
        try:
            connection.send({"error": f"{type(exc).__name__}: {exc}"})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


def _receive(connection, timeout: float):
    if not connection.poll(timeout):
        raise RuntimeError(f"Worker não respondeu em {timeout:g}s.")
    try:
        result = connection.recv()
    except (EOFError, OSError) as exc:
        raise RuntimeError("Worker encerrou antes de devolver a contribuição.") from exc
    if isinstance(result, dict) and "error" in result:
        raise RuntimeError(f"Falha no worker: {result['error']}")
    return result


def train_distributed(
    X: np.ndarray,
    y: np.ndarray,
    workers: int,
    epochs: int,
    learning_rate: float,
    *,
    start_method: str = "spawn",
    demo: bool = False,
    timings: dict | None = None,
    timeout: float = 60,
) -> tuple[np.ndarray, list[float], float]:
    """Driver coordena processos persistentes com uma partição fixa cada."""
    validate_training(X, y, epochs, learning_rate)
    if timeout <= 0 or not np.isfinite(timeout):
        raise ValueError("O timeout precisa ser finito e positivo.")
    start = time.perf_counter()
    parts = split_partitions(X, y, workers)
    partitioned = time.perf_counter()
    weights = np.zeros(X.shape[1] + 1, dtype=np.float64)
    history: list[float] = []
    context = mp.get_context(start_method)
    processes, connections, worker_info = [], [], []
    try:
        for partition_id, (xp, yp) in enumerate(parts):
            parent, child = context.Pipe()
            process = context.Process(
                target=_worker, args=(child, partition_id, xp, yp)
            )
            connections.append(parent)
            try:
                process.start()
            finally:
                child.close()
            processes.append(process)
        for connection in connections:
            worker_info.append(_receive(connection, timeout))
        ready = time.perf_counter()
        if demo:
            print(
                f"DRIVER pid={os.getpid()} | dados enviados uma vez; workers prontos",
                flush=True,
            )
        for epoch in range(epochs + 1):
            # BROADCAST simulado: todas as partições recebem os mesmos pesos.
            for connection in connections:
                connection.send(weights)
            local_results = [
                _receive(connection, timeout) for connection in connections
            ]
            # REDUCE: agrega somas de gradiente e loss e normaliza pelo total de exemplos.
            total_n = sum(n for n, _, _ in local_results)
            gradient_sum = sum(
                (gradient for _, gradient, _ in local_results),
                start=np.zeros_like(weights),
            )
            loss_sum = sum(loss for _, _, loss in local_results)
            history.append(loss_sum / total_n)
            global_gradient = gradient_sum / total_n
            if demo and epoch == 0:
                for info, (n, gradient, _) in zip(worker_info, local_results):
                    print(
                        f"MAP partição={info['partition']} pid={info['pid']} linhas={n} norma_gradiente_soma={np.linalg.norm(gradient):.6f}",
                        flush=True,
                    )
                print(
                    f"REDUCE driver={os.getpid()} linhas={total_n} norma_gradiente_médio={np.linalg.norm(global_gradient):.6f}",
                    flush=True,
                )
            if epoch < epochs:
                weights -= learning_rate * global_gradient
        finished = time.perf_counter()
        for connection in connections:
            connection.send(None)
        for process in processes:
            process.join(timeout)
            if process.is_alive() or process.exitcode != 0:
                raise RuntimeError("Worker não encerrou normalmente.")
    finally:
        # Inclusive falhas de inicialização ou cálculo: não deixar processos órfãos.
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join()
        for connection in connections:
            connection.close()
    total = time.perf_counter() - start
    if timings is not None:
        timings.update(
            partition_seconds=partitioned - start,
            startup_seconds=ready - partitioned,
            iteration_seconds=finished - ready,
            shutdown_seconds=total - (finished - start),
            total_seconds=total,
            workers=worker_info,
        )
    return weights, history, total


def prepare_data(
    path: Path,
    sample_size: int,
    seed: int,
    test_size: float,
    *,
    month: str = "2025-01",
    metadata: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    if (
        sample_size < 0
        or not np.isfinite(test_size)
        or not 0 < test_size < 1
        or seed < 0
    ):
        raise ValueError("Use sample-size >= 0, seed >= 0 e 0 < test-size < 1.")
    try:
        period = pd.Period(month, freq="M")
        if str(period) != month:
            raise ValueError("Formato incorreto")
    except (ValueError, TypeError) as exc:
        raise ValueError("Use month no formato AAAA-MM.") from exc
    columns = [
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "trip_distance",
        "fare_amount",
        "passenger_count",
        "tip_amount",
        "payment_type",
    ]
    frame = pd.read_parquet(path, columns=columns)
    raw_rows = len(frame)
    frame = frame.loc[frame.payment_type == 1, columns].copy()
    card_rows = len(frame)
    frame["tpep_pickup_datetime"] = pd.to_datetime(
        frame.tpep_pickup_datetime, errors="coerce"
    )
    frame["tpep_dropoff_datetime"] = pd.to_datetime(
        frame.tpep_dropoff_datetime, errors="coerce"
    )
    frame["duration_minutes"] = (
        frame.tpep_dropoff_datetime - frame.tpep_pickup_datetime
    ).dt.total_seconds() / 60
    frame["pickup_hour"] = frame.tpep_pickup_datetime.dt.hour
    for column in ["trip_distance", "fare_amount", "passenger_count", "tip_amount"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    # Cada descarte é contado uma vez, pelo primeiro filtro que reprova a linha.
    masks = {
        "timestamps": frame.tpep_pickup_datetime.notna()
        & frame.tpep_dropoff_datetime.notna(),
        "pickup_outside_month": frame.tpep_pickup_datetime.ge(period.start_time)
        & frame.tpep_pickup_datetime.lt((period + 1).start_time),
        "nonfinite_values": np.isfinite(
            frame[FEATURES + ["tip_amount"]].to_numpy(dtype=float)
        ).all(axis=1),
        "trip_distance": frame.trip_distance.between(0.1, 100),
        "fare_amount": frame.fare_amount.between(1, 500),
        "passenger_count": frame.passenger_count.between(1, 6),
        "duration_minutes": frame.duration_minutes.between(1, 240),
        "tip_amount": frame.tip_amount.ge(0),
    }
    valid = np.ones(len(frame), dtype=bool)
    discarded = {"not_credit_card": raw_rows - card_rows}
    for reason, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        discarded[reason] = int(np.sum(valid & ~mask))
        valid &= mask
    frame = frame.loc[valid, FEATURES + ["tip_amount"]]
    valid_rows = len(frame)
    if len(frame) < 2:
        raise ValueError("Menos de duas corridas válidas após o pré-processamento.")
    if sample_size > 0 and len(frame) > sample_size:
        frame = frame.sample(n=sample_size, random_state=seed)
    y = tip_labels(frame.tip_amount.to_numpy(), frame.fare_amount.to_numpy())
    X = frame[FEATURES].to_numpy(dtype=np.float64)
    counts = np.bincount(y.astype(int), minlength=2)
    test_rows = math.ceil(len(y) * test_size)
    if counts.min() < 2 or min(test_rows, len(y) - test_rows) < 2:
        raise ValueError(
            "Split estratificado exige ao menos duas linhas de cada classe e dois exemplos em treino e teste. Aumente a amostra ou ajuste test-size."
        )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    if len(np.unique(y_train)) != 2 or len(np.unique(y_test)) != 2:
        raise ValueError(
            "Uma classe ficou ausente no split. Aumente a amostra ou ajuste test-size."
        )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    if metadata is not None:
        metadata.update(
            raw_rows=raw_rows,
            card_rows=card_rows,
            valid_rows=valid_rows,
            discarded=discarded,
            sample_rows=len(y),
            train_rows=len(y_train),
            test_rows=len(y_test),
            month=month,
            features=FEATURES,
            label_rule="5 * round_even(tip_amount * 100) >= round_even(fare_amount * 100)",
            train_class_counts=np.bincount(y_train.astype(int), minlength=2).tolist(),
            test_class_counts=np.bincount(y_test.astype(int), minlength=2).tolist(),
            scaler_mean=scaler.mean_.tolist(),
            scaler_scale=scaler.scale_.tolist(),
        )
    return X_train, X_test, y_train, y_test


def classification_metrics(y: np.ndarray, probabilities: np.ndarray) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        log_loss,
        roc_auc_score,
    )

    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "confusion_matrix": confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
    }


def evaluate(weights: np.ndarray, X: np.ndarray, y: np.ndarray) -> dict:
    return classification_metrics(y, sigmoid(X @ weights[1:] + weights[0]))


def run_benchmark(
    X_train,
    X_test,
    y_train,
    y_test,
    *,
    workers,
    epochs,
    learning_rate,
    repeats,
    seed,
    start_method,
):
    """Mesmos dados e algoritmo; ordem embaralhada e comparação em toda repetição."""
    if repeats < 1 or not workers or len(set(workers)) != len(workers):
        raise ValueError("Use repeats >= 1 e uma lista de workers sem duplicatas.")
    if any(w < 1 or w > len(y_train) for w in workers):
        raise ValueError("Use 1 <= workers <= número de exemplos de treino.")
    reference, expected_loss, _ = train_sequential(
        X_train, y_train, epochs, learning_rate
    )
    configurations = [(1, "sequencial")] + [(n, "processos") for n in workers]
    runs = [[] for _ in configurations]
    first_outputs = {}
    orders = []
    rng = np.random.default_rng(seed)
    for repeat in range(repeats):
        order = rng.permutation(len(configurations)).tolist()
        orders.append([f"{configurations[i][1]}:{configurations[i][0]}" for i in order])
        for index in order:
            n, mode = configurations[index]
            timings = {}
            if mode == "sequencial":
                weights, history, _ = train_sequential(
                    X_train, y_train, epochs, learning_rate, timings=timings
                )
            else:
                weights, history, _ = train_distributed(
                    X_train,
                    y_train,
                    n,
                    epochs,
                    learning_rate,
                    start_method=start_method,
                    timings=timings,
                )
            np.testing.assert_allclose(weights, reference, rtol=1e-9, atol=1e-10)
            np.testing.assert_allclose(history, expected_loss, rtol=1e-10, atol=1e-12)
            runs[index].append(timings)
            first_outputs.setdefault(index, (weights, history))
            print(
                f"Repetição {repeat+1}/{repeats} | {mode} {n} | total={timings['total_seconds']:.3f}s | iterações={timings['iteration_seconds']:.3f}s",
                flush=True,
            )
    results = []
    timing_keys = [
        "partition_seconds",
        "startup_seconds",
        "iteration_seconds",
        "shutdown_seconds",
        "total_seconds",
    ]
    for index, (n, mode) in enumerate(configurations):
        weights, history = first_outputs[index]
        seconds = [run["total_seconds"] for run in runs[index]]
        iteration_seconds = [run["iteration_seconds"] for run in runs[index]]
        results.append(
            {
                "workers": n,
                "mode": mode,
                "seconds": float(np.median(seconds)),
                "q1_seconds": float(np.quantile(seconds, 0.25)),
                "q3_seconds": float(np.quantile(seconds, 0.75)),
                "iteration_q1_seconds": float(np.quantile(iteration_seconds, 0.25)),
                "iteration_q3_seconds": float(np.quantile(iteration_seconds, 0.75)),
                "timings_median": {
                    key: float(np.median([run[key] for run in runs[index]]))
                    for key in timing_keys
                },
                "runs": runs[index],
                "loss": history,
                "weights": weights.tolist(),
                **evaluate(weights, X_test, y_test),
            }
        )
    for result in results:
        result["speedup"] = results[0]["seconds"] / result["seconds"]
        result["iteration_speedup"] = (
            results[0]["timings_median"]["iteration_seconds"]
            / result["timings_median"]["iteration_seconds"]
        )
    prevalence = float(np.mean(y_train))
    # Classe é escolhida no TREINO. A probabilidade constante também vem dele.
    baseline = classification_metrics(y_test, np.full(len(y_test), prevalence))
    baseline.update(
        predicted_class=int(prevalence >= 0.5), train_positive_fraction=prevalence
    )
    return {
        "results": results,
        "majority_baseline": baseline,
        "execution_orders": orders,
        "equivalence_checked_every_run": True,
    }


def save_charts(results: list[dict], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [
        "Sequencial" if r["mode"] == "sequencial" else f"{r['workers']} processo(s)"
        for r in results
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, iteration in zip(axes, [False, True]):
        medians = np.array(
            [
                r["timings_median"]["iteration_seconds"] if iteration else r["seconds"]
                for r in results
            ]
        )
        prefix = "iteration_" if iteration else ""
        lower = np.array([r[prefix + "q1_seconds"] for r in results])
        upper = np.array([r[prefix + "q3_seconds"] for r in results])
        ax.bar(
            labels,
            medians,
            yerr=np.maximum(0, [medians - lower, upper - medians]),
            capsize=4,
            color="#3274a1",
        )
        ax.set_ylabel("Tempo (s): mediana e intervalo interquartil")
        ax.set_title(
            "Somente iterações"
            if iteration
            else "Total: partições, início, iterações e fim"
        )
        ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    fig.savefig(output_dir / "tempo_por_workers.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, result in zip(labels, results):
        ax.plot(
            range(len(result["loss"])),
            result["loss"],
            marker="o",
            markevery=max(1, len(result["loss"]) // 6),
            label=label,
            alpha=0.75,
        )
    ax.set_xlabel("Atualizações concluídas (0 = pesos iniciais)")
    ax.set_ylabel("Binary cross-entropy no treino")
    ax.set_title("Loss por estado do modelo: curvas equivalentes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_por_epoca.png", dpi=160)
    plt.close(fig)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment_metadata() -> dict:
    cpu = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in cpuinfo.read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu,
        )
    packages = [
        "numpy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "matplotlib",
        "threadpoolctl",
        "scipy",
    ]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "threadpools": threadpool_info(),
        "source_sha256": file_sha256(Path(__file__)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("data/yellow_tripdata_2025-01.parquet")
    )
    parser.add_argument(
        "--month",
        default="2025-01",
        help="Mês do pickup no formato AAAA-MM; dropoff pode cruzar o mês.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=300_000,
        help="0 usa todas as corridas válidas.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Processos a comparar; inclui controle com 1 worker. Sequencial é sempre incluído.",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--start-method", choices=mp.get_all_start_methods(), default="spawn"
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Mostrar uma época fora das medições; usa até 4 processos.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    if (
        args.epochs < 1
        or not np.isfinite(args.learning_rate)
        or args.learning_rate <= 0
    ):
        parser.error("Use epochs >= 1 e learning-rate finito > 0.")
    if (
        args.sample_size < 0
        or not np.isfinite(args.test_size)
        or not 0 < args.test_size < 1
        or not 0 <= args.seed < 2**32
    ):
        parser.error("Use sample-size >= 0, 0 < test-size < 1 e 0 <= seed < 2**32.")
    if (
        args.repeats < 1
        or any(w < 1 for w in args.workers)
        or len(set(args.workers)) != len(args.workers)
    ):
        parser.error("Use repeats >= 1 e workers >= 1 sem duplicatas.")
    if not args.data.is_file():
        parser.error(
            f"Parquet não encontrado: {args.data}. Consulte o download oficial no README."
        )
    # Filhos com spawn herdam estes limites antes de importar NumPy.
    for variable in [
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ]:
        os.environ[variable] = "1"
    with threadpool_limits(limits=1):
        data_info = {}
        data_start = time.perf_counter()
        try:
            X_train, X_test, y_train, y_test = prepare_data(
                args.data,
                args.sample_size,
                args.seed,
                args.test_size,
                month=args.month,
                metadata=data_info,
            )
            if max(args.workers) > len(y_train):
                raise ValueError(
                    "Número de workers excede o total de linhas de treino."
                )
        except ValueError as exc:
            parser.error(str(exc))
        data_info["preprocessing_seconds"] = time.perf_counter() - data_start
        data_info["input"] = {
            "path": str(args.data.resolve()),
            "bytes": args.data.stat().st_size,
            "sha256": file_sha256(args.data),
        }
        print(
            f"Treino: {len(y_train):,} | teste: {len(y_test):,} | positivas no treino: {y_train.mean():.2%}",
            flush=True,
        )
        if args.demo:
            print("DEMONSTRAÇÃO: uma atualização, fora do benchmark", flush=True)
            train_distributed(
                X_train,
                y_train,
                min(4, max(args.workers)),
                1,
                args.learning_rate,
                start_method=args.start_method,
                demo=True,
            )
        experiment = run_benchmark(
            X_train,
            X_test,
            y_train,
            y_test,
            workers=args.workers,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            repeats=args.repeats,
            seed=args.seed,
            start_method=args.start_method,
        )
        experiment.update(
            schema_version=2,
            sample_size_requested=args.sample_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
            test_size=args.test_size,
            repeats=args.repeats,
            start_method=args.start_method,
            threads_per_process=1,
            data=data_info,
            environment=environment_metadata(),
            timing_scope="total inclui particionamento, startup, epochs+1 avaliações de treino, atualizações e shutdown; exclui validação de entrada, leitura/preprocessamento, demonstração, referência sequencial e teste",
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        save_charts(experiment["results"], args.output_dir)
        (args.output_dir / "resultados.json").write_text(
            json.dumps(experiment, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        for result in experiment["results"]:
            print(
                f"{result['mode']} {result['workers']} | mediana={result['seconds']:.3f}s | speedup={result['speedup']:.2f}x | acc={result['accuracy']:.4f} | F1={result['f1']:.4f} | balanced_acc={result['balanced_accuracy']:.4f}"
            )
        print(f"Baseline majoritária: {experiment['majority_baseline']}")
        print(f"Resultados em: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
