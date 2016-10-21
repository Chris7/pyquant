import os
import six
# import plotly.plotly as py
# import plotly
# plotly.tools.set_credentials_file(username=os.environ.get('plotly_username'), api_key=os.environ.get('plotly_api_key'))
# from plotly.graph_objs import Scatter
from functools import wraps

def timer(func):
    @wraps(func)
    def inner(*args, **kwargs):
        import time
        start = time.time()
        val = func(*args, **kwargs)
        elapsed = time.time()-start
        print('{} evaluated in {}'.format(func.func_name if six.PY2 else func.__name__, elapsed))
        # trace = Scatter(
        #     x=os.environ.get('TRAVIS_BUILD_NUMBER', 1),
        #     y=elapsed,
        #     name=func.func_name
        # )
        # data = [trace]
        # py.plot(data, filename='pyquant_stats_{}'.format(func.func_name), fileopt='extend', auto_open=False)
        return val
    return inner