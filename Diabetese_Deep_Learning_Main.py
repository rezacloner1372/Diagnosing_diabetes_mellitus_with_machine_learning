# -------------------------------------------------------------------------------
# Name:        Diabetese_Deep_Learning_Main
# Purpose:     Diabetese diagnosisi with deep learning
# Author:      Saberi
# Created:     20 February 2023
# Licence:     licenced by Saberi
# -------------------------------------------------------------------------------
from Diabetese_Deep_Learning_Functions import *


def main():
    Diabetese_Positive_DataFrame = Load_Diabetese_DataFrame(
        Diabetese_Positive_Dataset, Diabetese_Positive_Worksheet)
    Diabetese_Negative_DataFrame = Load_Diabetese_DataFrame(
        Diabetese_Negative_Dataset, Diabetese_Negative_Worksheet)
    Diabetese_Positive_DataFrame = ImproveMissingData(
        Diabetese_Positive_DataFrame, True)
    Diabetese_Negative_DataFrame = ImproveMissingData(
        Diabetese_Negative_DataFrame, False)
    Diabetese_Positive_DataFrame = ImproveErrorData(
        Diabetese_Positive_DataFrame, True)
    Diabetese_Negative_DataFrame = ImproveErrorData(
        Diabetese_Negative_DataFrame, False)
    LunchCNN()


if __name__ == '__main__':
    main()
