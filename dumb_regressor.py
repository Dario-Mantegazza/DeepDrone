from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def dumb_regressor_result(x_test, x_train, y_test, y_train):
    """
          Dumb regressor, predict only the mean value for each target variable,
          returns MAE and MSE metrics per each variable.

          Args:
            x_test: validation samples
            x_train: training samples
            y_test: validation target
            y_train: training target

          Returns:
            dumb_metrics: list of metrics results after dumb regression
    """
    dumb_reg = DummyRegressor()
    fake_data = np.zeros((x_train.shape[0], 1))
    fake_test = np.zeros((1, 1))
    dumb_reg.fit(fake_data, y_train)
    dumb_pred = dumb_reg.predict(fake_test)[0]
    dumb_pred_x = np.full((x_test.shape[0], 1), dumb_pred[0])
    dumb_pred_y = np.full((x_test.shape[0], 1), dumb_pred[1])
    dumb_pred_z = np.full((x_test.shape[0], 1), dumb_pred[2])
    dumb_pred_yaw = np.full((x_test.shape[0], 1), dumb_pred[3])
    dumb_mse_x = mean_squared_error(y_test[:, 0], dumb_pred_x)
    dumb_mae_x = mean_absolute_error(y_test[:, 0], dumb_pred_x)
    dumb_mse_y = mean_squared_error(y_test[:, 1], dumb_pred_y)
    dumb_mae_y = mean_absolute_error(y_test[:, 1], dumb_pred_y)
    dumb_mse_z = mean_squared_error(y_test[:, 2], dumb_pred_z)
    dumb_mae_z = mean_absolute_error(y_test[:, 2], dumb_pred_z)
    dumb_mse_yaw = mean_squared_error(y_test[:, 3], dumb_pred_yaw)
    dumb_mae_yaw = mean_absolute_error(y_test[:, 3], dumb_pred_yaw)
    dumb_metrics = [[dumb_mse_x, dumb_mae_x], [dumb_mse_y, dumb_mae_y], [dumb_mse_z, dumb_mae_z], [dumb_mse_yaw, dumb_mae_yaw]]
    return dumb_metrics