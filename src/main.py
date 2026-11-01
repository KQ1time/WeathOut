# v.0.0.1
import joblib
import pandas as pd
from data import OUTFIT_DECODE
from train import MODEL_PATH

def get_data():
    while True:
        try:
            t = int(input("Temperature: "))
            if not -25 <= t <= 31:
                print("Input a correct temperature!")
                continue
        except ValueError:
            print("Input a correct temperature!")
            continue
        try:
            wind_speed = int(input("Wind speed: "))
            if not 0 <= wind_speed <= 20:
                print("Input a correct wind speed!")
                continue
        except ValueError:
            print("Input a correct wind speed!")
            continue
        try:
            humidity = int(input("Humidity: "))
            if not 0 <= humidity <= 100:
                print("Input a correct humidity!")
                continue
        except ValueError:
            print("Input a correct himidity!")
            continue
        try:
            rain = int(input("Rain (0 or 1): "))
            if rain != 0 and rain != 1:
                print("Inpus a correct rain!")
                continue
        except ValueError:
            print("Input a correct rain")
            continue
        return (t, wind_speed, humidity, rain)


def main():
    t, wind_speed, humidity, rain = get_data()

    data = {"temperature": t, "wind_speed": wind_speed, "humidity": humidity, "rain": rain}

    model = joblib.load(MODEL_PATH)
    result = model.predict(pd.DataFrame([data]))
    
    print(OUTFIT_DECODE[result[0]])

if __name__ == "__main__":
    main()
