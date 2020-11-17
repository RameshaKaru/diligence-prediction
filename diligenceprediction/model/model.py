from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input, Softmax


def cmlstmModel(num_rules=18, lstm_units=50, num_months=6, dense_param=[2]):
    """
    Model with cmeans labels

    Parameters
    ----------
    num_rules : int
        number of rules/ number of non-diligence probabilities per ANM per time frame
    lstm_units : int
        number of lstm units
    num_months :  int
        number of months in non-diligence vector history taken as input
    dense_param : int
        dense layer parameter

    Returns
    -------
    model : compiled model
    """

    first_input = Input(shape=(num_months, num_rules))
    second_lstm = LSTM(lstm_units)(first_input)
    third_dense = Dense(dense_param[0], activation="relu")(second_lstm)
    fourth_softmax = Softmax()(third_dense)

    model = Model(inputs=first_input, outputs=fourth_softmax)

    model.compile(optimizer='adam', loss='mse',
                  metrics=['mse', 'mae'])

    return model


def normModel(num_rules=18, lstm_units=50, num_months=6, dense_param=[2, 1]):
    """
      Model with simple norm score labels

      Parameters
      ----------
      num_rules : int
          number of rules/ number of non-diligence probabilities per ANM per timeframe
      lstm_units : int
          number of lstm units
      num_months :  int
          number of months in non-diligence vector history taken as input
      dense_param : int
          dense layer parameter

      Returns
      -------
      model : compiled model
    """

    first_input = Input(shape=(num_months, num_rules))
    second_lstm = LSTM(lstm_units)(first_input)
    third_dense = Dense(dense_param[0], activation="relu")(second_lstm)
    fourth_dense = Dense(dense_param[1])(third_dense)

    model = Model(inputs=first_input, outputs=fourth_dense)

    model.compile(optimizer='adam', loss='mse',
                  metrics=['mse', 'mae'])

    return model


