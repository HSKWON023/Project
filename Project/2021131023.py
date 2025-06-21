class ModuloCalculator:
    def __init__(self, n):
        self.n = n
        self.elements = list(range(n))

    def add(self, a, b):
        return (a + b) % self.n

    def mul(self, a, b):
        return (a * b) % self.n

    def inverse(self, a):
        for i in range(1, self.n):
            if (a * i) % self.n == 1:
                return i
        return None

    def is_unit(self, a):
        return self.inverse(a) is not None

    def zero_divisors(self):
        result = []
        for a in range(1, self.n):
            for b in range(1, self.n):
                if (a * b) % self.n == 0 and a != 0 and b != 0 and a != b:
                    result.append((a, b))
        return result

    def unit_group(self):
        return [a for a in self.elements if self.is_unit(a)]

def main():
    n = int(input("모듈 연산에 사용할 n값을 입력하세요 (예: 12): "))
    calc = ModuloCalculator(n)

    while True:
        print("\n1. 덧셈\n2. 곱셈\n3. 곱셈 역원 구하기\n4. 유니트 원소들\n5. 영인자 목록\n6. 종료")
        choice = input("원하는 작업을 선택하세요: ")

        if choice == "1":
            a, b = map(int, input("두 수를 입력하세요 (공백으로 구분): ").split())
            print(f"{a} + {b} ≡ {calc.add(a, b)} (mod {n})")

        elif choice == "2":
            a, b = map(int, input("두 수를 입력하세요 (공백으로 구분): ").split())
            print(f"{a} × {b} ≡ {calc.mul(a, b)} (mod {n})")

        elif choice == "3":
            a = int(input("역원을 구할 수를 입력하세요: "))
            inv = calc.inverse(a)
            if inv is None:
                print(f"{a}의 역원은 존재하지 않습니다.")
            else:
                print(f"{a}의 역원은 {inv}입니다. {a} × {inv} ≡ 1 (mod {n})")

        elif choice == "4":
            print(f"유니트 원소: {calc.unit_group()}")

        elif choice == "5":
            print(f"영인자 목록: {calc.zero_divisors()}")

        elif choice == "6":
            print("프로그램을 종료합니다.")
            break

        else:
            print("잘못된 선택입니다.")

if __name__ == "__main__":
    main()
