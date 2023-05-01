from sklearn.metrics import mean_squared_error, mean_absolute_error , r2_score
from smogn import phi,phi_ctrl_pts
import pandas as pd
import numpy as np

def calculate_phi(y, method = "auto", xtrm_type = "both", coef = 1.5, ctrl_pts = None):
    """
    Calculates the phi value for each element of 'y'.
    
    Parameters
    ----------
    y : array-like
        Input data for which phi value needs to be calculated.
    method : str, optional (default='auto')
        Relevance method. Either 'auto' or 'manual'.
    xtrm_type : str, optional (default='both')
        Distribution focus. Either 'high', 'low', or 'both'.
    coef : float, optional (default=1.5)
        Coefficient for box plot (pos real)
    ctrl_pts : list, optional (default=None)
        Input for "manual" rel method  (2d array).
    
    Returns
    -------
    y_phi : array-like
        Phi values for each element of 'y'.
    
    """
    
    if not isinstance(y, pd.DataFrame):
        y = pd.core.series.Series(y)
    try:    
        phi_params = phi_ctrl_pts(y = y,method = method, xtrm_type = xtrm_type, coef = coef, ctrl_pts = ctrl_pts)
    except Exception as e:
        raise Exception(e)
    else:
        try: 
            y_phi = phi(y = y,ctrl_pts = phi_params)
        except Exception as e:
            raise Exception(e)
        else:
            return y_phi
        



def phi_weighted_r2(y, y_pred, method = "auto", xtrm_type = "both", coef = 1.5, ctrl_pts = None):
    """
    Calculates the R^2 score between 'y' and 'y_pred' with weighting by phi.
    
    Parameters
    ----------
    y : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    method : str, optional (default='auto')
        Relevance method. Either 'auto' or 'manual'.
    xtrm_type : str, optional (default='both')
        Distribution focus. Either 'high', 'low', or 'both'.
    coef : float, optional (default=1.5)
        Coefficient for box plot (pos real)
    ctrl_pts : list, optional (default=None)
        Input for "manual" rel method  (2d array).

    Returns
    -------
    r2 : float
        Phi weighted R^2 score.
    
    """

    y_phi=calculate_phi(y, method, xtrm_type , coef, ctrl_pts)
    return r2_score(y, y_pred, sample_weight=y_phi)


def phi_weighted_mse(y, y_pred, method = "auto", xtrm_type = "both", coef = 1.5, ctrl_pts = None):
    """
    Calculates the mean squared error between 'y' and 'y_pred' with weighting by phi.
    
    Parameters
    ----------
    y : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    method : str, optional (default='auto')
        Relevance method. Either 'auto' or 'manual'.
    xtrm_type : str, optional (default='both')
        Distribution focus. Either 'high', 'low', or 'both'.
    coef : float, optional (default=1.5)
        Coefficient for box plot (pos real)
    ctrl_pts : list, optional (default=None)
        Input for "manual" rel method  (2d array).
        
    Returns
    -------
    mse : float
        Phi weighted Mean squared error.
    
    """

    y_phi=calculate_phi(y, method, xtrm_type , coef, ctrl_pts)
    return mean_squared_error(y, y_pred, sample_weight=y_phi)


def phi_weighted_mae(y, y_pred, method = "auto", xtrm_type = "both", coef = 1.5, ctrl_pts = None):

    """
    Calculates the mean absolute error between 'y' and 'y_pred' with weighting by phi.
    
    Parameters
    ----------
    y : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    method : str, optional (default='auto')
        Relevance method. Either 'auto' or 'manual'.
    xtrm_type : str, optional (default='both')
        Distribution focus. Either 'high', 'low', or 'both'.
    coef : float, optional (default=1.5)
        Coefficient for box plot (pos real)
    ctrl_pts : list, optional (default=None)
        Input for "manual" rel method  (2d array).
        
    Returns
    -------
    phi_weighted_mae : float
       Phi weighted Mean absolute error.
    
    """

    y_phi=calculate_phi(y, method, xtrm_type , coef, ctrl_pts)
    return mean_absolute_error(y, y_pred, sample_weight=y_phi)


def phi_weighted_root_mse(y, y_pred, method = "auto", xtrm_type = "both", coef = 1.5, ctrl_pts = None):
    """
    Calculates the root mean squared error between 'y' and 'y_pred' with weighting by phi.
    
    Parameters
    ----------
    y : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    method : str, optional (default='auto')
        Relevance method. Either 'auto' or 'manual'.
    xtrm_type : str, optional (default='both')
        Distribution focus. Either 'high', 'low', or 'both'.
    coef : float, optional (default=1.5)
        Coefficient for box plot (pos real)
    ctrl_pts : list, optional (default=None)
        Input for "manual" rel method  (2d array).
        
    Returns
    -------
    mae : float
        Phi weighted Root Mean squared error.
    """

    y_phi=calculate_phi(y, method, xtrm_type , coef, ctrl_pts)
    return np.sqrt(mean_squared_error(y, y_pred, sample_weight=y_phi))



def ser_process(trues, preds, method = "auto", xtrm_type = "both", coef = 1.5, ctrl_pts = None):
    
    if not isinstance(preds, pd.DataFrame):
        preds = pd.core.series.Series(preds)
    if not isinstance(trues, pd.DataFrame):
        trues = pd.core.series.Series(trues)
    
    trues=trues.reset_index(drop=True)
    phi_trues= calculate_phi(trues, method, xtrm_type , coef, ctrl_pts) 

    #trues = trues.values
    tbl = pd.DataFrame(
        {'trues': trues,
         'phi_trues': phi_trues,
         })
    tbl = pd.concat([tbl, preds], axis=1)
    ms = list(tbl.columns[2:])
    return tbl,ms

def ser_t(y, y_pred, t, method = "auto", xtrm_type = "both", coef = 1.5,  ctrl_pts = None, s=0):
    """
    Calculates the Squared error-relevance values between 'y' and 'y_pred' with weighting by phi at thershold 't'.
    
    Parameters
    ----------
    y : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    t : float
        Threshold value.
    method : str, optional (default='auto')
        Relevance method. Either 'auto' or 'manual'.
    xtrm_type : str, optional (default='both')
        Distribution focus. Either 'high', 'low', or 'both'.
    coef : float, optional (default=1.5)
        Coefficient for box plot (pos real)
    ctrl_pts : list, optional (default=None)
        Input for "manual" rel method  (2d array).
        
    Returns
    -------
    ser_t : list
        List of Squared error-relevance values at each threshold t.
    """
    tbl,ms= ser_process(y, y_pred, method, xtrm_type , coef , ctrl_pts)

    error = [sum(tbl.apply(lambda x: ((x['trues'] - x[y]) ** 2) if x['phi_trues'] >= t else 0, axis=1)) for y in ms]
    
    if s==1:
        return error
    else:
        return error[0]


def sera (y, y_pred, step = 0.01,return_err = False, method = "auto", xtrm_type = "both", coef = 1.5, ctrl_pts = None, weight= None) :

    """
    Calculates the Squared error-relevance areas (ERA) between y and y_pred.

    Parameters
    ----------
    y : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    step : float, optional (default=0.001)
        Step size for threshold values
    return_err : bool, optional (default=False)
        Whether to return the error and thershold values with the SERA value.
    method : str, optional (default='auto')
        Relevance method. Either 'auto' or 'manual'.
    xtrm_type : str, optional (default='both')
        Distribution focus. Either 'high', 'low', or 'both'.
    coef : float, optional (default=1.5)
        Coefficient for box plot (pos real)
    ctrl_pts : list, optional (default=None)
        Input for "manual" rel method  (2d array).
        
    Returns
    -------
    sera : float or dict:
        If `return_err` is False, returns the SERA value as a float. If `return_err` is True, returns a dictionary containing the SERA value, the error values, and the thresholds used in the calculation.

    """
    _,ms= ser_process(y, y_pred)
    th = np.arange(0, 1 + step, step)
    errors = []
    for ind in th:
        errors.append(ser_t(y, y_pred, ind, method , xtrm_type , coef , ctrl_pts , s=1))
        

    areas = []
    for x in range(1, len(th)):
        areas.append([step *(errors[x - 1][y] + errors[x][y]) / 2 for y in range(len(ms))])
    areas = pd.DataFrame(data=areas, columns=ms)
    res = areas.apply(lambda x: sum(x))
    if return_err :
       return {"sera":res, "errors":[item for sublist in errors for item in sublist], "thrs" :th}
    else:
       return res.item()