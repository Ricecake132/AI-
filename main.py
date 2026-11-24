"""Mini NPU Simulator (Python 3.8+)."""
import json
import math
import re
from pathlib import Path
from time import perf_counter_ns

EPSILON = 1e-9
REPEATS = 100
SIZES = (3, 5, 13, 25)


def mac(pattern, kernel):
    total = 0.0
    for row, weights in zip(pattern, kernel):
        for value, weight in zip(row, weights):
            total += value * weight
    return total


def decide(a, b, labels):
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError("MAC 점수가 유한수 범위를 벗어났습니다.")
    if abs(a - b) < EPSILON:
        return "UNDECIDED"
    return labels[0] if a > b else labels[1]


def normalize(label):
    labels = {"+": "Cross", "cross": "Cross", "x": "X"}
    if isinstance(label, str) and label.strip().lower() in labels:
        return labels[label.strip().lower()]
    raise ValueError("알 수 없는 라벨 {!r}".format(label))


def matrix(data, n):
    if not isinstance(data, list) or len(data) != n:
        raise ValueError("행 개수 오류 (필요한 행 수 {})".format(n))
    result = []
    for index, row in enumerate(data, 1):
        if not isinstance(row, list) or len(row) != n:
            raise ValueError("{}행 열 개수 오류 (필요한 열 수 {})".format(index, n))
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in row):
            raise ValueError("{}행에 유한한 숫자가 필요합니다.".format(index))
        result.append([float(v) for v in row])
    return result


def read_matrix(name):
    print(name, "(3줄, 공백 구분)")
    rows = []
    while len(rows) < 3:
        tokens = input("{}행> ".format(len(rows) + 1)).split()
        try:
            row = [float(token) for token in tokens]
            if len(row) == 3 and all(math.isfinite(v) for v in row):
                rows.append(row)
                continue
        except ValueError:
            pass
        print("입력 형식 오류: 이 행에 유한한 숫자 3개를 공백으로 구분해 다시 입력하세요.")
    return rows


def generated_patterns(n):
    mid = n // 2
    cross = [[float(i == mid or j == mid) for j in range(n)] for i in range(n)]
    diagonal = [[float(i == j or i + j == n - 1) for j in range(n)] for i in range(n)]
    return cross, diagonal


def benchmark(samples):
    print("\n성능 분석 (필터 1개, 크기별 {}회 평균)".format(REPEATS))
    print("크기       평균 시간(ms)    MAC 횟수(N²)")
    for n, (pattern, kernel) in samples.items():
        elapsed = 0
        for _ in range(REPEATS):
            start = perf_counter_ns()
            mac(pattern, kernel)
            elapsed += perf_counter_ns() - start
        print("{0}×{0}\t{1:.6f}\t{2}".format(n, elapsed / REPEATS / 1e6, n*n))


def manual():
    a = read_matrix("필터 A")
    b = read_matrix("필터 B")
    print("필터 A, B 저장 완료")
    pattern = read_matrix("패턴")
    start = perf_counter_ns()
    first, second = mac(pattern, a), mac(pattern, b)
    elapsed = (perf_counter_ns() - start) / 1e6
    print("A 점수: {} | B 점수: {}".format(first, second))
    print("두 필터 연산 시간: {:.6f} ms".format(elapsed))
    result = decide(first, second, ("A", "B"))
    print("판정:", "판정 불가 (UNDECIDED)" if result == "UNDECIDED" else result)
    benchmark({3: (pattern, a)})


def load_filters(data):
    filters, errors = {}, {}
    for n in SIZES[1:]:
        key = "size_{}".format(n)
        try:
            pair = data.get(key) if isinstance(data, dict) else None
            if not isinstance(pair, dict):
                raise ValueError("필터 묶음 누락 또는 형식 오류")
            normalized = {}
            for raw, values in pair.items():
                label = normalize(raw)
                if label in normalized:
                    raise ValueError("중복 필터 " + label)
                normalized[label] = matrix(values, n)
            if set(normalized) != {"Cross", "X"}:
                raise ValueError("Cross/X 필터가 모두 필요합니다.")
            filters[key] = normalized
            print(key, "필터 로드 완료 (Cross, X)")
        except (ValueError, OverflowError) as err:
            errors[key] = str(err)
            print(key, "필터 오류:", err)
    return filters, errors


def batch(path):
    with path.open(encoding="utf-8-sig") as stream:
        data = json.load(stream)
    if not isinstance(data, dict) or not isinstance(data.get("patterns"), dict):
        raise ValueError("patterns 객체가 필요합니다. 테스트를 집계할 수 없습니다.")
    filters, errors = load_filters(data.get("filters"))
    failures = {}
    for key, case in data["patterns"].items():
        print("\n[{}]".format(key))
        try:
            match = re.fullmatch(r"(size_([1-9][0-9]*))_[0-9]+", key)
            if not match:
                raise ValueError("패턴 키 형식 오류 (size_N_idx 필요)")
            group = match[1]
            if group not in filters:
                raise ValueError(group + ": " + errors.get(group, "대응 필터 없음"))
            if not isinstance(case, dict):
                raise ValueError("패턴 항목은 객체여야 합니다.")
            expected = normalize(case.get("expected"))
            pattern = matrix(case.get("input"), int(match[2]))
            pair = filters[group]
            a, b = mac(pattern, pair["Cross"]), mac(pattern, pair["X"])
            print("Cross 점수: {} | X 점수: {}".format(a, b))
            result = decide(a, b, ("Cross", "X"))
            print("판정: {} | expected: {}".format(result, expected), end=" | ")
            if result != expected:
                reason = "허용오차 내 동점" if result == "UNDECIDED" else "점수 순위와 라벨 불일치"
                raise ValueError("{} (판정 {}, expected {})".format(reason, result, expected))
            print("PASS")
        except (ValueError, OverflowError) as err:
            failures[key] = str(err)
            print("FAIL:", err)
    benchmark({n: generated_patterns(n) for n in SIZES})
    total = len(data["patterns"])
    print("\n총 테스트: {} | 통과: {} | 실패: {}".format(total, total-len(failures), len(failures)))
    for key, reason in failures.items():
        print("-", key, reason)


def main():
    print("=== Mini NPU Simulator ===")
    try:
        while True:
            choice = input("1. 사용자 입력 (3×3)\n2. data.json 분석\n선택> ").strip()
            if choice == "1":
                manual()
                break
            if choice == "2":
                batch(Path(__file__).with_name("data.json"))
                break
            print("1 또는 2를 입력하세요.")
    except EOFError:
        print("\n입력이 끝나 실행을 종료합니다. 필요한 행을 모두 입력해 주세요.")
    except (ValueError, OSError) as err:
        print("실행 오류:", err)


if __name__ == "__main__":
    main()
