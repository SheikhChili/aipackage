from setuptools import setup, find_packages

setup(
    name='aipackage',
    version='1.0',
    license='MIT',
    author="SheikhChili",
    author_email='email@example.com',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/SheikhChili/aipackage',
    keywords='ML Automation',
    install_requires=[
        'scikit-learn', 'xgboost', 'catboost', 'lightgbm', 'numpy', 'pandas', 'tensorflow', 'imblearn', 'optuna',
        'tpot', 'hyperopt', 'yellowbrick', 'sys', 'time', 'threading', 'csv', 'pickle', 'os', 'librosa', 'math',
        'keras_tuner', 'json', 're', 'spacy', 'unidecode', 'bs4', 'random', 'warnings', 'dtale', 'lime', 'matplotlib',
        'shap', 'sweetviz', 'aix360', 'explainerdashboard', 'pandas_profiling', 'pandas_visual_analysis'
    ],

)
