from setuptools import setup, find_packages

setup(
    name='ai_automation',
    version='1.0',
    license='MIT',
    author="SheikhChili",
    author_email='sparrow.shreyas@gmail.com',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/SheikhChili/aipackage',
    keywords='ML Automation',
    install_requires=[
        'scikit-learn', 'xgboost', 'catboost', 'lightgbm', 'numpy', 'pandas', 'tensorflow', 'imblearn', 'optuna',
        'tpot', 'hyperopt', 'yellowbrick', 'librosa', 'keras_tuner', 'spacy', 'unidecode', 'bs4', 'dtale',
        'lime', 'matplotlib', 'shap', 'sweetviz', 'aix360', 'explainerdashboard', 'pandas_profiling',
        'pandas_visual_analysis'
    ],

)
