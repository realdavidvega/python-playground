# Recap of Python concepts used in langgraph

if __name__ == "__main__":
    # Dict
    # key-value pairs
    person_dict = {"name": "John Doe", "age": 30, "city": "New York"}

    # TypedDict
    # type safety and enhanced readability
    from typing import TypedDict, ReadOnly

    class Person(TypedDict):
        name: ReadOnly[str]  # readonly field
        age: int
        weight: float
        height: float
        city: str

    person_typed: Person = {
        "name": "John Doe",
        "age": 30,
        "weight": 70.5,
        "height": 177.8,
        "city": "New York",
    }

    print(f"Person name is {person_typed['name']}\n")

    # Union
    # flexible types and easier to code
    # type safety with hints
    from typing import Union

    def get_imc(weight: Union[float, int], height: Union[float, int]):
        return weight / (height**2)

    imc_float = get_imc(70.5, 177.8)  # ok
    imc_int = get_imc(70, 177)  # ok
    # imc = get_imc("70", "177.8") # error

    # Optional
    # can be None
    from typing import Optional

    def print_person(person: Optional[Person]) -> None:
        if person is not None:
            print(
                f"Hello {person['name']}, you currently weight {person['weight']}kg and are {person['height']}cm tall\n"
            )
        else:
            print("Hello, how can I help you?\n")

    print_person(person_typed)

    # Any
    # can be anything
    from typing import Any

    def print_hero(hero: Any) -> None:
        print(hero)

    print_hero("Sometimes I feel like a hero\n")

    # Lambda functions
    # to create anonymous functions, inline functions, higher-order functions, closures, etc.

    # ruff: noqa: E731
    imc_lambda = lambda weight, height: weight / (height**2)
    print(f"My IMC is {imc_lambda(70.5, 177.8)}\n")

    weights = [70.2, 69.5, 67.1]
    heights = [177.8, 177.8, 177.8]
    imcs = list(map(imc_lambda, weights, heights))
    print(f"My IMCs are {imcs}\n")

    imcs_alt = [imc_lambda(weight, height) for weight, height in zip(weights, heights)]
    print(f"My IMCs are {imcs_alt}\n")
