from setuptools import setup, find_packages

setup(
    name='ai_automation',
    version='2.7.0',
    license='MIT',
    author="SheikhChili",
    author_email='sparrow.shreyas@gmail.com',
    packages=['aiautomation', 'aiautomation/mlpackage/classification', 'aiautomation/mlpackage',
              'aiautomation/mlpackage/multiclassification', 'aiautomation/mlpackage/regression',
              'aiautomation/mlpackage/segmentation'],
    # package_dir={'': 'src'},
    url='https://github.com/SheikhChili/aipackage',
    keywords='AI Automation',
    install_requires=[
        'scikit-learn', 'xgboost', 'catboost', 'lightgbm', 'numpy', 'pandas', 'tensorflow', 'imblearn', 'optuna',
        'tpot', 'hyperopt', 'yellowbrick', 'librosa', 'keras_tuner', 'spacy', 'unidecode', 'bs4', 'dtale',
        'lime', 'matplotlib', 'shap', 'sweetviz', 'aix360', 'explainerdashboard', 'pandas_profiling',
        'pandas_visual_analysis'
    ],

)
