# -------------------------------------------------------------------------------
# Name:        Diabetese_Deep_Learning_Functions
# Purpose:     Diabetese diagnosisi with deep learning
# Author:      Saberi
# Created:     20 February 2023
# Licence:     licenced by Saberi
# -------------------------------------------------------------------------------
from Diabetese_Deep_Learning_Include_Library import *


def PlotShowTime(Plot, Interval):
    # set the timer interval 5000 milliseconds
    timer = Plot.gcf().canvas.new_timer(interval=Interval)
    timer.add_callback(Plot.close)
    timer.start()
    Plot.show()


def MaximiseWindowOfFigure(Plot):
    Plot.gcf()
    figManager = Plot.get_current_fig_manager()
    figManager.window.showMaximized()


def Load_Diabetese_DataFrame(Diabetese_Dataset, Diabetese_Worksheet):
    Diabetese_DataFrame = pd.read_excel(Diabetese_Dataset, Diabetese_Worksheet)
    return Diabetese_DataFrame


def ImproveMissingData(Old_Diabetese_DataFrame, Index):
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    imputer.fit(Old_Diabetese_DataFrame)
    x = imputer.transform(Old_Diabetese_DataFrame)
    New_Diabetese_DataFrame = pd.DataFrame(
        x, columns=Old_Diabetese_DataFrame.columns)
    if Index:
        print('Diabetese remove Missing Data',
              'Info: Any missing data of Diabetese Positive Dataset is removed.')
        PrintSeparator(50)
    else:
        print('Diabetese remove Missing Data',
              'Info: Any missing data of Diabetese Negative Dataset is removed.')
        PrintSeparator(50)
    return New_Diabetese_DataFrame


def ImproveErrorData(Old_Diabetese_DataFrame, Index):
    X = Old_Diabetese_DataFrame.to_numpy()
    X_Normalized = Normalizer().fit(X)
    X_Normalized = Normalizer().transform(X)
    New_Diabetese_DataFrame = pd.DataFrame(
        X_Normalized, columns=Old_Diabetese_DataFrame.columns)
    if Index:
        print('Diabetese correct Error Data',
              'Info: Any error data of Diabetese Positive Dataset is corrected.')
        PrintSeparator(50)
    else:
        print('Diabetese correct Error Data',
              'Info: Any error data of Negative Positive Dataset is corrected.')
        PrintSeparator(50)
    return New_Diabetese_DataFrame


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def Split(X: np.ndarray, Y: np.ndarray, trS: float, vaS: float):
    trX, X2, trY, Y2 = train_test_split(X, Y, train_size=trS)
    teS = 1 - trS - vaS
    vaX, teX, vaY, teY = train_test_split(X2, Y2, train_size=vaS/(vaS+teS))
    Output = {'X': {'tr': trX,
                    'va': vaX,
                    'te': teX},
              'Y': {'tr': trY,
                    'va': vaY,
                    'te': teY}}
    return Output


def ConvertDataSetToImage():
    df_shape_01 = pd.read_excel(
        Diabetese_Positive_Dataset, sheet_name=Diabetese_Positive_Worksheet)
    df_shape_02 = pd.read_excel(
        Diabetese_Negative_Dataset, sheet_name=Diabetese_Negative_Worksheet)
    Matrix_Shape_02 = np.array(df_shape_02)
    Matrix_Shape_01 = np.array(df_shape_01.values.tolist())[
        0:len(Matrix_Shape_02)]
    ECG_Dataset_Array = np.concatenate((Matrix_Shape_01, Matrix_Shape_02))
    M = Diabetese_Dataset_Array[:, 0].shape
    N = Diabetese_Dataset_Array[0, :].shape
    scaled_x = NormalizeData(Diabetese_Dataset_Array)
    ImageList = []
    ImageArray = np.zeros([sum(M), sum(N), 3], dtype=np.uint8)
    for i in range(0, sum(scaled_x[:, 0].shape)):
        for j in range(0, sum(scaled_x[0, :].shape)):
            ImageArray[i, j, :] = int(scaled_x[i, j] * 255)
            ImageList.append(int(scaled_x[i, j] * 255))
    L = ImageArray[0, 0, :].shape
    import random
    for k in range(0, sum(L)):
        ImageArray[0, 0, k] = random.random() * 255
    plt.imshow(ImageArray)
    MaximiseWindowOfFigure(plt)
    PlotShowTime(plt, Interval)
    print(M)
    from PIL import Image
    img = Image.fromarray(ImageArray)
    img.save('Diabetese.png')
# Function to create model, required for KerasClassifier


def create_model():
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


def LunchCNN():
    df_shape_01 = pd.read_excel(
        Diabetese_Positive_Dataset, sheet_name=Diabetese_Positive_Worksheet)
    PrintSeparator(50)
    print('Diabetese Positive Dataset DataFrame is:\n', df_shape_01)
    PrintSeparator(50)
    df_shape_01.plot(colormap='jet')
    plt.title('Diabetese Positive Dataset DataFrame')
    plt.xlabel('Index')
    plt.ylabel('Amount')
    MaximiseWindowOfFigure(plt)
    PlotShowTime(plt, Interval)
    df_shape_02 = pd.read_excel(
        Diabetese_Negative_Dataset, sheet_name=Diabetese_Negative_Worksheet)
    print('Diabetese Negative Dataset DataFrame is:\n', df_shape_02)
    PrintSeparator(50)
    df_shape_02.plot(colormap='jet')
    plt.title('Diabetese Negative Dataset DataFrame')
    plt.xlabel('Index')
    plt.ylabel('Amount')
    MaximiseWindowOfFigure(plt)
    PlotShowTime(plt, Interval)
    df_shape_03 = pd.concat([df_shape_01, df_shape_02], axis=0)
    print('Diabetese Combination Positive & Negative Dataset DataFrame is:\n', df_shape_03)
    PrintSeparator(50)
    df_shape_03.plot(colormap='jet')
    plt.title('Diabetese Combination of Positive & Negative Dataset DataFrame')
    plt.xlabel('Index')
    plt.ylabel('Amount')
    MaximiseWindowOfFigure(plt)
    PlotShowTime(plt, Interval)
    Matrix_Shape_02 = np.array(df_shape_02)
    Matrix_Shape_01 = np.array(df_shape_01.values.tolist())[
        0:len(Matrix_Shape_02)]
    X = Matrix_Shape_01
    Y = Matrix_Shape_02
    Output = Split(X, Y, 0.8, 0.15)
    trX = Output['X']['tr']
    print('Train X is:\n', trX)
    PrintSeparator(50)
    plt.plot(trX)
    plt.set_cmap('jet')
    plt.legend(labels=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
               'Insulin', 'BMI', 'DiabetesPedigreeFunction    Age', 'class'])
    plt.title('Train X')
    plt.xlabel('Index')
    plt.ylabel('Amount')
    MaximiseWindowOfFigure(plt)
    PlotShowTime(plt, Interval)
    trY = Output['Y']['tr']
    print('Train Y is:\n', trY)
    PrintSeparator(50)
    plt.plot(trY)
    plt.legend(labels=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
               'Insulin', 'BMI', 'DiabetesPedigreeFunction    Age', 'class'])
    plt.title('Train Y')
    plt.set_cmap('jet')
    plt.xlabel('Index')
    plt.ylabel('Amount')
    MaximiseWindowOfFigure(plt)
    PlotShowTime(plt, Interval)
    teX = Output['X']['te']
    print('Test X is:\n', teX)
    PrintSeparator(50)
    plt.plot(teX)
    plt.legend(labels=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
               'Insulin', 'BMI', 'DiabetesPedigreeFunction    Age', 'class'])
    plt.title('Test X')
    plt.set_cmap('jet')
    plt.xlabel('Index')
    plt.ylabel('Amount')
    MaximiseWindowOfFigure(plt)
    PlotShowTime(plt, Interval)
    teY = Output['Y']['te']
    print('Test Y is:\n', teY)
    PrintSeparator(50)
    plt.plot(teY)
    plt.legend(labels=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
               'Insulin', 'BMI', 'DiabetesPedigreeFunction    Age', 'class'])
    plt.title('Test Y')
    plt.set_cmap('jet')
    plt.xlabel('Index')
    plt.ylabel('Amount')
    MaximiseWindowOfFigure(plt)
    PlotShowTime(plt, Interval)
    # Step 1
    # Importing the necessary libraries doing in Diabetese_Deep_Learning_Include_Library
    # Step 2
    # Loading the dataset
    dataset = load_digits()
    print(' Result of Load Digit function is:\n:', dataset)
    PrintSeparator(50)
    # df_shape_03
    # Step 3
    # Splitting the data into tst and train
    # 80 - 20 Split
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.20, random_state=4)
    # Step 4
    # Making the Neural Network Classifier & Set Hidden layer size and its attributes
    Neural_Network = MLPClassifier(hidden_layer_sizes=(
        150, 100, 50), max_iter=300, activation='relu', solver='adam')
    print('Result of classifier of Neural Network is:\n', Neural_Network)
    PrintSeparator(50)
    # Step 5
    # Training the model on the training data and labels
    Neural_Network.fit(x_train, y_train)
    PrintSeparator(50)
    print('Result of X Train is:\n', x_train)
    PrintSeparator(50)
    print('Result of Y Train is:\n', y_train)
    PrintSeparator(50)
    # Step 6
    # load the Diabetese_Dataset
    Diabetese_Dataset = loadtxt(Diabetese_CSV_Dataset, delimiter=',')
    # split into input (X) and output (y) variables
    X = Diabetese_Dataset[:, 0:8]
    y = Diabetese_Dataset[:, 8]
    # define the keras model
    model = Sequential()
    model.add(Dense(12, input_shape=(8,), activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile the keras model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    # fit the keras model on the Diabetese_Dataset
    model.fit(X, y, epochs=150, batch_size=10)
    # evaluate the keras model
    _, accuracy = model.evaluate(X, y)
    print('Accuracy: %.2f' % (accuracy*100))
    # Step 7
    # Testing the model i.e. predicting the labels of the test data in new approach
    y_pred = Neural_Network.predict(x_test)
    print('Result of Prediction is:\n', y_pred)
    PrintSeparator(50)
    # Step 7
    # Evaluating the results of the model
    Accuracy = accuracy_score(y_test, y_pred) * 100
    Precision = precision_score(y_test, y_pred, average=None) * 100
    Confusion_Matrix = confusion_matrix(y_test, y_pred)
    # Step 8
    # Printing the Results
    print("Accuracy for Neural Network is:\n", Accuracy)
    PrintSeparator(50)
    print("Precision for Neural Network is:\n", Precision)
    PrintSeparator(50)
    print("Average of Precision for Neural Network is:\n", Average(Precision))
    PrintSeparator(50)
    print("Confusion Matrix is:\n", Confusion_Matrix)


def PrintSeparator(Range):
    Separator = ''
    for i in range(0, Range):
        Separator = Separator + '-'
    print(Separator)


def Average(MyList):
    Sum = 0
    for i in range(len(MyList)):
        Sum = Sum + MyList[i]
    AverageSum = Sum / len(MyList)
    return AverageSum
