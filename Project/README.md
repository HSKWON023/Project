
# 주제 : 선형점화식의 일반항 유도 및 그래프 시각화 프로젝트

                                                            2021131016 이준원

## 1. 모티베이션(프로젝트를 하게 된 동기)

   : 수학과 프로그래밍 수업 진행간, Recursion(점화식)의 코딩 방식(Memorization, Bottom up)에 대해서 배울 수 있었다. 해당 방식을 통해, 기존 점화식 작성 코드보다 더욱 빠르고 적은 용량으로 점화식의 값을 구할 수 있었다. 하지만, 해당 코딩 방식에서는 "매우 큰 항에 대해서는 값을 구하는 것이 다소 어렵다는 점, 점화식을 일반화하기 어렵다는 점(항이 3개, 4개, ...등 인 경우, 내가 원하는 설정(점화식의 계수, 초기항의 값)으로 점화식을 구하기 어렵다는 점 등"이 문제점으로 느껴졌다. 이산수학 과목에서 배운 내용에 따르면, 특성방정식을 만들고 해당 방정식의 근을 구함으로써, 선형점화식의 일반항을 계산할 수 있다는 것을 배웠다. 이를 접목하여, Top Down(Native, Memorization), Bottom up 방식이 아닌 새로운 방식을 사용해 선형점화식을 일반화하면 maximum recursion에 제약받지 않고 매우 큰 항의 값도 빠른 시간내에 구할 수 있을 것이라 판단하여 해당 주제로 프로젝트를 진행하였다.

## 2. 이론적 배경

  1) 선형 점화식 및 특성 방정식
     : 수열 ${a_n}$에 대하여 $a_{n+k}+c_1a_{n+k-1}+...+c_ka_n=f(n)$의 형태로 나타내는 점화식을 선형 점화식이라고 한다. 이때, r에 대한 방정식 $r^{n+k}+c_1r^{n+k-1}+...+c_kr{n}=0$을 이 점화식의 특성방정식이라 한다. 

  2) 선형점화식의 일반항 유도
     : 1)의 특성방정식에서, $α_{1}, α_{2}, ..., α_{n}$을 특성방정식의 근이라 하자. 이때, 점화식의 일반항은 $α_{1}^n, α_{2}^n, ..., α_{n}^n$의 일차결합으로 표현된다.
     <br/>(1) 만약 특성방정식의 근에서 중복된 근이 존재하지 않으면, ${a_n}=A_1α_{1}^n, A_2α_{2}^n, ..., A_nα_{n}^n$으로 나타낼 수 있으며, 초기항($a_1, a_2, ...$)의 값을 대입하여, $A_1, A_2, ..., A_n$을 구할 수 있다.
    <br/>(2) 만약 특성벙장석의 근에서 어떤 근 β가 k번 중복된다면, 중복되는 근은 $β^n, nβ^n, ... , n^{k-1}β^n$으로 표현해야 한다. 이후, 중복되지 않은 근과의 일차결합을 통해 특성방정식의 일반항을 표현할 수 있다.

## 3. 코드 작성방법 및 설명

  ### 1) import
```
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import sys
```
1. numpy : 점화식의 일반항을 유도하고, 이를 그래프로 옮기는 코드를 작성하는 과정에서, np.arange을 이용해 x축에 변수를 추가해주었다.

2. matplotlib.pyplot : 점화식의 일반항을 그래프로 그리는 과정에서 matplotlib 모듈을 사용하였다.

3. sympy : 변수를 지정하고, 방정식을 설정하고, 방정식의 해를 구하기 위해 해당 모듈을 사용하였다.
   
   (a) `x = sp.symbols('n')` : x를 기호변수 n으로 선언해준다. 해당 기호변수를 통해, 수식을 나타내고 이에 대한 근을 구하거나, 기호변수에 특정한 값을 대입할 수 있다.
   
   (b) `sp.Eq(LHS, RHS)` : 좌변에는 LHS, 우변에는 RHS으로 이루어진 등식을 만들어준다. (ex. `E = sp.Eq(x**2, 1)` => `E`: `x**2 = 1` (`E`라는 변수에 `x**2 = 1`을 대입한다.)
   
   (c) `sp.Rational(a, b)` : 유리수를 분수 {a/b}로 나타낸다. `sp.Rational(a)`는 `a`로 출력된다.
   
   (d) `sp.solve(Eqn, x)`, `sp.roots(Eqn, x)` : `Eqn`이라는 방정식에 대하여 `x`에 관한 해를 구한다. `sp.solve(Eqn, x)`는 `x`에 관한 해만 출력하는 반면, `sp.roots(Eqn, x)`는 다항식의 해를 keys, 각 해의 중복도를 values로 갖는 dictionary를 반환한다.
   
   (e) `sp.lambdify(x, fct, 'numpy')` : `x`라는 변수를 갖는 `fct` 표현식을 실제 계산에 사용할 수 있도록 파이썬 함수로 변환할 수 있다. (Ex. `f = sp.lambdify(x, sp.sin(x), 'numpy')`로 지정한다면, `f(np.pi/2)`는 1이라는 값을 갖는다.

   (f) eqn.subs(x, 1) : 기호변수에 특정한 값을 대입하거나, 다른 기호변수로 대체할 수 있는 메소드이다.
   <br/> (ex. eqn= $x^2+5$  => eqn.subs(x,2)= $2^2+5=9$ / eqn.subs(x,t)= $t^2+5$

5. sys : `sys.exit()`코드를 이용해, 프로젝트의 코드를 종료하게 만들었다.

본격적인 설명에 앞서, 정의한 메소드, 함수들은 RecurrenceRelation 클래스 안에 있다.
  ### 2) __init__
  ```
    def __init__(self): #변수 저장
        self.coefficients = [] # (e.g, Fn=2*F(n-1)+3*F(n-2) => self.coefficients = [2,3])
        self.initial_values = [] # (e.g, F1=1, F2=2 => self.initial_values = [1,2])
        self.char_eq_roots = [] # 특성방정식의 해를 저장하는 variable
        self.general_solution = None # 점화식의 일반항을 저장하는 variab
```

이후 후술할 메소드들에서 구한 데이터들을 각 변수들에 반환한다.

self.coefficients : 선형점화식에서 이루어진 계수들을 저장하는 변수이다.
<br/>(Ex. $F_n=2F_{n-1}+3F_{n-2}$의 구조로 이루어진 선형점화식을 데이터로 갖는다면, self.coefficients = [2, 3]이 된다.)

self.initial_values : 선형점화식에서 정해진 초기항들의 값들을 저장하는 변수이다.
<br/>(Ex, $F_n=2F_{n-1}+3F_{n-2}$ / F1=2, F2=5 라면, self.initial_values=[2, 5]이다.

self.char_eq_roots : 특성방정식의 근과 중복도를 각각 keys, values로 갖는 dictionary를 저장한다.
<br/>(Ex, $F_n=-2F_{n-1}-1F_{n-2}$에서, 특성방정식의 근은 -1, 중복도는 2이다. 이때, self.char_eq_roots={-1 : 2}가 된다.

self.general_solution : 점화식의 일반항을 유도한 뒤, 해당 일반항을 저장한다.

  ### 3) set_variables
  ```
    def set_variables(self): #1번 변수 저장 함수 : 점화식 계산에 필요한 변수개수, 항의 계수, 점화식의 초기항 등을 입력하고 저장한다
      try:
        n = int(input("변수의 개수를 입력하세요 (ex. 2 => Fn=Fn-1+Fn-2): "))
        self.coefficients = [sp.Rational(input(f"계수 a{i+1}을(를) 입력하세요: ")) for i in range(n)]
        self.initial_values = [sp.Rational(input(f"F{i+1}을(를) 입력하세요: ")) for i in range(n)]
        print("변수가 성공적으로 지정되었습니다.")
```

main 함수에서의 1번 함수이다. (코드 실행 후, 1을 입력 시 해당 함수가 실행된다.)

위 함수에서, 점화식을 설정하고 각 데이터들을 변수에 반환하는 기능을 한다.

처음 n은 항의 개수이다. 구하고자 하는 선형점화식의 항의 개수를 입력하면 된다. 
<br/>(ex. n=3이면, $F_n=a_1F_{n-1}+a_2F{n-2}+a_3F{n_3}$꼴의 선형점화식을 가짐. n=2이면, $F_n=a_1F_{n-1}+a_2F{n-2}$꼴의 선형점화식을 가짐.)

이후, 각 점화식의 계수와 초기항의 값들을 입력한다.
<br/>(ex. $n=2, a_1=1, a_2=1, F_1=1, F_2=1$으로 설정하면, $F_n=F_{n-1}+F_{n-2} / F_1=1, F_2=1$(피보나치 수열)의 선형점화식으로 설정 가능)

  ### 4) compute_characteristic_equation
  ```
    def compute_characteristic_equation(self): #2번 특성방정식 및 해 계산 함수 : 1번 변수 저장 함수를 통해 저장한 변수를 이용해, 해당 점화식의 특성방정식을 구하고 이를 계산한다.
        if self.coefficients==[]:
            print("변수 지정을 먼저 해주세요.")
            return

        x = sp.symbols('x')
        n = len(self.coefficients)
        LHS = x**n - sum(self.coefficients[i] * x**(n-1-i) for i in range(n)) #특성방정식에서의 좌항(LHS) 계산
        equation = sp.Eq(LHS, 0) #좌항=LHS, 우항=0

        self.char_eq_roots = sp.roots(equation, x) #좌항=LHS, 우항=0 인 방정식의 해 계산 (이때, self.char_eq_roots는 해를 keys, 중복도를 values로 갖는 딕셔너리)

        print(f"특성방정식: \n{LHS} = 0")
        print(f"특성방정식의 해: {self.char_eq_roots}")
```

main 함수에서의 2번 함수이다. (코드 실행 후, 2를 입력 시 해당 함수가 실행된다.)

1번 함수에서 지정한 선형점화식 변수들을 이용해서 특성방정식을 유도하고 특성방정식의 해 및 중복도를 계산하는 함수이이다.

만약 1번 함수를 실행하지 않고, 2번 함수를 실행할 시, 설정한 선형점화식이 없기 때문에 "변수 지정을 먼저 해주세요."라는 문구와 함께 return된다.

위 코드에서, x는 기호변수 x로 작동한다. LHS는 특성방정식의 좌항으로, 설정한 선형점화식을 변수 x에 관한 식으로 변환한 뒤, 이들을 전부 좌항으로 이항한 값이다.
<br/>(ex. $F_n=F_{n-1}+F_{n-2} (피보나치 수열)$이라면, $x^n=x^{n-1}+x^{n-2}$가 되고, LHS는 이들을 전부 좌항으로 이항한 $x^n-x^{n-1}-x^{n-2}$가 된다.)

따라서, 위와 같은 식으로 LHS 변수를 지정했다.

이에 따라, 우항은 0이 되므로 특성방정식은 sp.Eq(LHS,0)이 된다.

이후, sp.roots를 통해 해당 방정식을 계산하고 특성방정식의 해와 중복도를 각각 keys와 values로 갖는 dictionary를 self.char_eq_roots에 저장하고 특성방정식과 해당 dictionary를 출력하였다.

  ### 5) compute_general_term
```
    def compute_general_term(self): #3번 점화식의 일반항 계산 함수 : 2번 특성방정식 및 해 계산  함수를 통해 나온 해(근)을 통해, 점화식의 일반항을 도출한다.
      if self.char_eq_roots==[]:
        print("특성방정식의 해를 먼저 계산해주세요.")
        return

      k = sp.symbols('n')
      terms=[] #일반항은 각각의 계수*terms의 원소 들의 합으로 표현할 수 있다. 이때, terms는 특성방정식의 해와 중복도를 이용함
      for a, b in self.char_eq_roots.items():
        if b==1: #중복도가 1인 경우, 해당 근에 n제곱한 뒤, terms에 추가한다.
          terms.append(a**k)
        elif b!=1: #중복도가 1보다 큰 경우, 해당 근에 n제곱한 뒤, b의 값만큼 n제곱을 한 뒤 각각을 terms에 추가한다.
          for j in range(b):
            terms.append(k**j * a**k)

      x = sp.symbols(f'x:{len(terms)}') 

      equation = sum(x[j] * terms[j] for j in range(len(terms))) #일반항은 각각의 계수*terms의 원소들의 합이다.
      equations = []

      for i in range(1, len(self.initial_values)+1):
        equations.append(sp.Eq(equation.subs(k,i),self.initial_values[i-1])) #위 equation에 각각의 초기항 대입하면 이들이 초기항의 데이터값과 같다. 이에 대한 연립방정식 

      constants=sp.solve(equations) #연립방정식을 풀면, 각각의 계수가 나온다. 이때, constants는 x0, x1,...,를 keys, 각 계수의 값을 values로 갖는 딕셔너리이다.
      general_solution_terms=[]
      for term, constant in zip(terms, constants.values()):
        general_solution_terms.append(constant * term)

      self.general_solution = sum(general_solution_terms)

      print("점화식의 일반항 계산 결과:")
      print(f"Fn = {self.general_solution}")
```

main 함수에서의 3번 함수이다. (코드 실행 후, 3을 입력 시 해당 함수가 실행된다.)

2번 함수에서 얻은 특성방정식의 근과 중복도, 1번 함수에서 얻은 점화식의 초기항의 값을 이용하여 점화식의 일반항을 유도하는 함수이다.

만약 2번 함수를 실행하지 않고, 3번 함수를 실행할 시, 특성방정식의 해와 중복도에 대한 데이터가 없기 때문에 "특성방정식의 해를 먼저 계산해주세요."라는 문구와 함께 return된다.

우선, 특성방정식의 해의 n제곱에 관한 일차결합으로 나타내기 전에, terms라는 list에 각 특성방정식의 해의 n제곱에 해당하는 데이터들을 넣어주었다.

이론적 배경에 의해, 중복도가 1이라면, 해당 근의 n제곱을 terms라는 list의 데이터로 append시켰다. 중복도가 1보다 크다면, 해당 근의 n제곱에 중복도의 크기만큼 n제곱한 각 데이터들을 terms라는 list의 데이터로 append시켰다.
<br/>(ex. 해 : 2 / 중복도 : 1 => $2^n$ 을 list의 원소로 append 시킴 | 해 : 3 / 중복도 : 3 => $3^n$, $n * 3^n$, $n^2 * 3^n$ 을 list의 원소로 append 시킴)

이후, x에 $x_0, x_1, ..., x_{n-1}$ 이라는 기호변수 데이터를 넣어주고 x와 terms의 곱을 더해주어 특성방정식의 해의 n제곱에 관한 일차결합을 나타내주었다. 이를 equation이라는 변수에 대입하였다.

이후에 해당 equation에서 n을 각각 1,2,...를 대입한 데이터들을 equations에 넣어주었다.

이제 equations를 좌항, 해당 선형점화식 초기항 데이터 값을 우항으로 갖는 연립방정식을 세운 뒤, 이를 sp.solve로 풀어주면 $x_0, x_1, ..., x_{n-1}$의 변수에 대한 데이터 값을 얻을 수 있다. 연립방정식을 만들 때, 우항에선 F1, F2, ....이 각각 self.initial_values[0], [1]에 할당되어 있으므로, self.initial_values[i-1]을 우항으로 설정하였다.

이렇게 구한 $x$를 constants라는 변수에 넣어주었다. constants는 각 $x_n$을 keys, $x_n$의 값을 values로 갖는 dictionary이다.

이제 constants.values()와 terms의 곱을 전부 더해주어 선형점화식의 일반항에 대한 값을 얻을 수 있었다. 또한, 해당 점화식의 일반항을 출력해주었다.

  ### 6) compute_specific_term
  ```
    def compute_specific_term(self): #4번 점화식의 값 대입 함수 : 3번에서 구한 점화식의 일반항을 통해, 점화식의 n번째 항에 해당하는 값을 계산한다.
      if not self.general_solution:
        print("점화식의 일반항을 먼저 계산해주세요.")
        return

      try:
        n = int(input("계산 결과를 구하고 싶은 항을 입력하세요: "))
        if n<=0:
          print("잘못된 입력입니다. 다시 시도해주세요")
          return
        else:
          specific_term = self.general_solution.subs(sp.symbols('n'),n).evalf(n=15) 
          print(f"F{n}의 값은 {specific_term}입니다.")

      except ValueError:
        print("잘못된 입력입니다. 다시 시도해주세요.")
```
main 함수에서의 4번 함수이다. (코드 실행 후, 4를 입력 시 해당 함수가 실행된다.)

3번 함수에서 유도한 점화식의 일반항을 토대로, 해당 값에 직접 원하는 데이터를 대입하여 해당 항의 값을 얻고 출력하는 함수이다.

만약 3번 함수를 실행하지 않고, 4번 함수를 실행할 시, 점화식의 일반항에 대한 데이터가 없기 때문에 "점화식의 일반항을 먼저 계산해주세요."라는 문구와 함께 return된다.

또한, 점화식의 초기항을 F1부터 시작했으므로, n<=0인 경우 "잘못된 입력입니다. 다시 시도해주세요"라는 문구와 함께 return된다.

계산 결과를 구하고 싶은 항을 입력하면, subs()매소드를 통해, n에 관한 점화식의 일반항에 해당 값을 직접 대입하고 이를 specific_term에 저장한다. 이때, evalf(n=15) 메소드를 이용해, 해당 항의 계산 결과 값을 15자리로 계산해준다.

이후, 해당 데이터값을 직접 출력하게된다.

  ### 7) plot_solution
  ```
    def plot_solution(self): # 3번에서 구한 점화식의 일반항을 그래프로 나타낸다.
        if not self.general_solution:
            print("점화식의 일반항을 먼저 계산해주세요.")
            return

        try:
            x = int(input("그래프로 항의 인덱스를 입력하세요: "))
        except ValueError:
            print("잘못된 입력입니다. 다시 시도해주세요.")
            return

        k = sp.symbols('n')
        general_solution_func = sp.lambdify(k, self.general_solution, 'numpy')
        n_values = np.arange(1, x + 1)
        general_solution_values = general_solution_func(n_values)

        plt.plot(n_values, general_solution_values, label='General Solution')
        plt.xlabel('n')
        plt.ylabel('F(n)')
        plt.title('Comparison of General Solution and Actual Recurrence')
        plt.legend()
        plt.show()
```

main 함수에서의 5번 함수이다. (코드 실행 후, 5를 입력 시 해당 함수가 실행된다.)

3번 함수에서 얻은 점화식의 일반항을 그래프로 시각화하여 해당 점화식이 어떠한 경향성을 띄고 있는지 확인할 수 있는 함수이다.

만약 3번 함수를 실행하지 않고, 5번 함수를 실행할 시, 점화식의 일반항에 대한 데이터가 없기 때문에 "점화식의 일반항을 먼저 계산해주세요."라는 문구와 함께 return된다.

표현하고자 하는 그래프의 인덱스 값을 입력하게 된다.(ex. 15입력 시, F1, ..., F15에 대한 데이터값을 그래프로 시각화해줌) 이후에, sp.lambdify를 이용해 구한 선형점화식의 일반항을 함수로 만들어 주고 np.arange를 통해 x축의 값을 만든 다음, 각 데이터값들을 함수에 input했을 때의 output 데이터를 general_solution_values라는 변수에 넣어주었다. 이후, x축의 데이터와 y축의 데이터를 이용해서 유도해낸 점화식의 일반항 결과 값을 그래프로 시각화해주었다.

  ### 8) main
  ```
def main(): # main 함수 : 1~6까지의 수를 입력하여 입력값에 해당하는 함수를 실행한다.
    rr = RecurrenceRelation()
    actions = {
        '1': rr.set_variables,
        '2': rr.compute_characteristic_equation,
        '3': rr.compute_general_term,
        '4': rr.compute_specific_term,
        '5': rr.plot_solution
    }

    while True:
        print("\n1. 변수 지정\n2. 점화식의 특성방정식 및 해 계산\n3. 점화식의 일반항 계산\n4. 점화식 대입\n5. 점화식의 그래프 시각화\n6. 종료")
        choice = input("실행할 함수를 선택하세요: ")
        if choice in actions:
            actions[choice]()
        elif choice=='6': #6번을 입력할 시, 코드를 종료한다.
          sys.exit("코드를 종료합니다")
        else:
            print("유효하지 않은 선택입니다. 다시 시도해주세요.")

if __name__ == "__main__":
    main()
```
해당 코드의 main 함수로, 6을 입력하기 전까지 해당 코드가 계속하여 작동하도록 해주었다.

1을 입력 시, set_variables 함수가 작동하여 선형점화식의 일반항을 구하는데 필요한 항의 개수, 점화식의 항의 계수, 점화식의 초기항 데이터 값을 입력하고 저장한다.

2를 입력 시, compute_characteristic_equation 함수가 작동하여 1번 함수에서 저장한 데이터를 바탕으로 특성방정식을 세운 뒤, 이에 대한 근과 중복도를 계산한다.

3을 입력 시, compute_general_term 함수가 작동하여 2번 함수에서 구한 특성방정식의 근과 중복도와 1번 함수에서 저장한 점화식의 초기항 데이터 값을 토대로 선형점화식의 일반항을 유도해낸다.

4를 입력시, compute_specific_term 함수가 작동하여 3번 함수에서 유도한 선형점화식의 일반항에 직접 원하는 항을 대입하여 해당 항의 선형점화식 데이터 값을 구할 수 있다.

5를 입력 시, plot_solution 함수가 작동하여 3번 함수에서 유도한 선형점화식의 일반항을 그래프로 시각화 할 수 있다.

6을 입력 시, sys.exit()이 작동하여 코드를 종료한다.

## 4. 선형점화식의 일반항 유도 제시 및 고찰

프로젝트를 통해, 기존 Top Down(Native, Memorization), Bottom up 방식이 아닌 특성방정식을 이용해 선형점화식의 일반항을 직접 유도하고 해당 일반항에 직접 값을 대입하여 원하는 점화식의 데이터 값을 출력해낼 수 있었다. 또한, 설정한 점화식을 그래프로 시각화할 수 있었다.

이번 프로젝트를 통해, 선형점화식의 일반항 유도 및 계산 프로젝트의 이점과 한계에 대해서 말하고자 한다.

 ### 1. 프로젝트의 이점

1) 새로운 방식으로 선형점화식의 특정 항에서의 값 계산 가능
  - 기존 Top down, Bottom up 방식에서 벗어나 새로운 방식으로 점화식의 특정 항의 값을 도출해낼 수 있었다. 실제로, 피보나치 수열($n=2, a_1=1, a_2=1, F_1=1, F_2=1$)의 몇가지 항에서 위 프로젝트의 결과값들을 실제 피보나치 수열의 값과 비교해 보았다.
- - -
피보나치 수열($n=2, a_1=1, a_2=1, F_1=1, F_2=1$)에서 5번째 항의 값 비교 (좌 : 프로젝트 결과값 / 우 : 실제 점화식 결과값)
<br/><img width="500" alt="F5" src="https://github.com/SungchulLee/probability/assets/170498248/0b567782-36bf-45c8-ba7e-344fb434f144">
<img width="300" alt="실제 F5" src="https://github.com/SungchulLee/probability/assets/170498248/07f129b3-0f0a-4af0-9a33-3f1c9bfe7a24">
- - -
- - -
피보나치 수열($n=2, a_1=1, a_2=1, F_1=1, F_2=1$)에서 35번째 항의 값 비교 (좌 : 프로젝트 결과값 / 우 : 실제 점화식 결과값)
<br/><img width="500" alt="F35" src="https://github.com/SungchulLee/probability/assets/170498248/31e14e6d-7082-4d75-81a4-a5ed84fc671f">
<img width="300" alt="F35 실제" src="https://github.com/SungchulLee/probability/assets/170498248/4cd4847e-b582-462c-aa4b-e7e1db0ea3a3">
- - -
**이를 통해, 프로젝트를 통해 얻은 결과값은 정밀하다는 것을 알 수 있었다. 또한, 특성방정식이 중근을 가지는 수열($n=2, a_1=-2, a_2=-1, F_1=1, F_2=1$)의 몇가지 항에서도 실제 점화식의 결과값과 비교해 보았다.**
- - -
특성방정식이 중근을 가지는 수열($n=2, a_1=-2, a_2=-1, F_1=1, F_2=1$)에 5번째 항의 비교
<br/><img width="400" alt="-2 -1 1 1 중근 F5" src="https://github.com/SungchulLee/probability/assets/170498248/405b706c-4450-4e16-b356-c99c005c430f">
<br/>실제 점화식의 값 : $F_1=1, F_2=1, F_3=-3, F_4=5, F_5=-7$
- - -
- - -
특성방정식이 중근을 가지는 수열($n=2, a_1=-2, a_2=-1, F_1=1, F_2=1$)에 10번째 항의 비교
<br/><img width="400" alt="-2 -1 1 1 중근 F10" src="https://github.com/SungchulLee/probability/assets/170498248/9bb46477-ad13-4203-a749-0ae5975fcfa1">
<br/>실제 점화식의 값 : $F_6=9, F_7=-11, F_8=13, F_9=-15, F_{10}=17$
- - -
  이를 통해, 프로젝트를 통해 얻은 일반항과 계산 결과값이 정밀하다는 것을 알 수 있다. 따라서, 기존의 Top down, Bottom up 방식이 아닌 새로운 방식으로 선형점화식에서 특정 항에서의 계산 결과값을 도출해낼 수 있었다
  
2) 일반화된 구조의 선형점화식에 대하여 결과 도출 가능
  - 해당 코드에선 피보나치 수열 등 특정 선형점화식 뿐만 아니라, 항의 개수, 점화식의 계수, 점화식의 초기항의 제한에서 국한되지 않는다는 점이 이번 프로젝트의 두번째 이점이다. 이전 방식에서는 피보나치 수열과 같은 특정 점화식에서만 사용할 수 있는 방식을 주로 사용했다. 또한, 항의 개수나 점화식의 계수 등은 고정되어 있었다. 하지만, 이번 프로젝트의 코드에서는 주어진 데이터를 바꾸고 싶을 때 마음대로 바꾸어 다양한 항의 개수, 점화식의 계수, 초기항의 값 등에서의 점화식의 구조를 마음대로 설정할 수 있었다. 또한, 설정한 점화식의 몇번째 항에서 무슨 값을 가지고 있는지를 확인할 수 있었다.

3) Maximum recursion의 제약에서 벗어남
   - 이전 코드들의 방식에서는 Maximum recursion의 문제점이 발생하여 매우 큰 n의 값에 대해서 해당 항에서의 값을 계산하는 것이 어려웠다. (아래참조 / 피보나치 수열)
  <br/><img width="400" alt="F1500000 실제" src="https://github.com/SungchulLee/probability/assets/170498248/ac51746d-999c-4317-ade9-ae5379be62b7">

   - 하지만, 해당 프로젝트를 통해, Maximum recursion의 제약에서 벗어나 매우 큰 n에 대해서도 해당 항의 값을 계산하는 것이 가능해졌으며, 매우 큰 값에 대해서도 빠른 속도로 계산이 가능해졌다. (아래 참조 / 피보나치 수열)
<br/><img width="600" alt="F1500000" src="https://github.com/SungchulLee/probability/assets/170498248/e796931b-9dd9-43cf-85e7-d1b8990f0419">

   - 점화식의 특정방정식을 이용한 일반항 유도 방식을 통해, Maximum recursion의 한계에서 벗어나 자유롭게 특정항이 가지는 값을 계산할 수 있게 됐다.
  
    ### 2. 프로젝트의 한계
  
1) 일반항 유도가 어려운 점화식의 경우, 일반항 계산 과정에서 불필요한 시간 소요
  - 일반항 유도가 어려운 점화식의 경우, 일반항 유도 과정에서 시간이 오래 소요되고 이 경우 오히려 효율이 떨어진다는 것을 알 수 있었다.
  - 예를 들어, Tribonacci Sequence의 경우가 이에 해당한다. Tribonacci Sequence란, $F_n=F_{n-1}+F_{n-2}+F_{n-3}$ | $F_1=1, F_2=1, F_3=2$를 만족시키는 점화식이다. 해당 점화식의 일반항은 아래와 같이 유도된다.
<br/><img width="450" alt="tribonacci" src="https://github.com/SungchulLee/probability/assets/170498248/3d21d4d2-a919-4a03-b75c-6e395dc5e77d">

  - 이렇듯, 일반항 유도 과정이 어려운 점화식은 일반항 계산 처리 과정에서 짧지않은 시간이 소요돼서 이러한 경우에선 효율이 떨어지게 된다.
```
점화식의 일반항 계산 결과:
Fn = (1/3 + (-1/2 - sqrt(3)*I/2)*(sqrt(33)/9 + 19/27)**(1/3) + 4/(9*(-1/2 - sqrt(3)*I/2)*(sqrt(33)/9 + 19/27)**(1/3)))**n*(-(3*sqrt(33) + 19)**(1/3)/16 - (3*sqrt(33) + 19)**(2/3)/64 + sqrt(33)*(3*sqrt(33) + 19)**(2/3)/2112 + 5*sqrt(33)*(3*sqrt(33) + 19)**(1/3)/528 - sqrt(3)*I*(3*sqrt(33) + 19)**(1/3)/16 - sqrt(11)*I*(3*sqrt(33) + 19)**(2/3)/704 + sqrt(3)*I*(3*sqrt(33) + 19)**(2/3)/64 + 5*sqrt(11)*I*(3*sqrt(33) + 19)**(1/3)/176) + (1/3 + 4/(9*(-1/2 + sqrt(3)*I/2)*(sqrt(33)/9 + 19/27)**(1/3)) + (-1/2 + sqrt(3)*I/2)*(sqrt(33)/9 + 19/27)**(1/3))**n*(-(3*sqrt(33) + 19)**(1/3)/16 - (3*sqrt(33) + 19)**(2/3)/64 + sqrt(33)*(3*sqrt(33) + 19)**(2/3)/2112 + 5*sqrt(33)*(3*sqrt(33) + 19)**(1/3)/528 - 5*sqrt(11)*I*(3*sqrt(33) + 19)**(1/3)/176 - sqrt(3)*I*(3*sqrt(33) + 19)**(2/3)/64 + sqrt(11)*I*(3*sqrt(33) + 19)**(2/3)/704 + sqrt(3)*I*(3*sqrt(33) + 19)**(1/3)/16) + (1/3 + 4/(9*(sqrt(33)/9 + 19/27)**(1/3)) + (sqrt(33)/9 + 19/27)**(1/3))**n*(-5*sqrt(33)*(3*sqrt(33) + 19)**(1/3)/264 - sqrt(33)*(3*sqrt(33) + 19)**(2/3)/1056 + (3*sqrt(33) + 19)**(2/3)/32 + (3*sqrt(33) + 19)**(1/3)/8)

계산 결과를 구하고 싶은 항을 입력하세요: 6
F6의 값은 13.0 + 0.e-22*I입니다.

계산 결과를 구하고 싶은 항을 입력하세요: 10
F10의 값은 149.0 + 0.e-22*I입니다.

(※참고 : F_1=1, F_2=1, F_3=2, F_4=4, F_5=7, F_6=13, F_7=24, F_8=44, F_9=81, F_10=149

계산 결과를 구하고 싶은 항을 입력하세요: 1500
F1500의 값은 3.16810272299545e+396 + 0.e-219*I입니다.
```

## 5. 결론
기존 파이썬의 Top down, Bottom up 방식이 아닌 특성방정식을 유도하고 이에 대한 해를 구하여 점화식의 일반항을 구하는 새로운 방식을 통해 점화식의 특정 항의 계산을 하였다. 위 프로젝트를 통해, 점화식을 새로운 방식으로 코딩할 수 있었다. 또한, 주어진 제약(항의 개수, 점화식의 계수, 초기항의 값)에서 벗어나 내가 원하는 점화식의 결과 데이터를 조건만 입력하면 얻을 수 있었다. 또한, 기존 방식에서의 문제점이었던 Maximum recursion에서 벗어나서 매우 큰 값에 대해서도 점화식의 특정 항에서의 값들을 빠른 시간에 얻어낼 수 있었다. 하지만, Tribonacci Sequence와 같이 일반항 유도 과정이 어려운 점화식의 경우, 일반항 유도에서 오랜 시간이 걸려 효율이 떨어진다는 것을 알 수 있었다.
