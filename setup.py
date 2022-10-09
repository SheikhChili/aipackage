from setuptools import setup

setup(
    name='ai_automation',
    version='3.0.1',
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
        'scikit-learn', 'xgboost', 'catboost', 'lightgbm', 'numpy', 'pandas==1.4.2', 'tensorflow', 'imblearn', 'optuna',
        'tpot', 'hyperopt', 'yellowbrick', 'librosa', 'keras_tuner', 'spacy', 'unidecode', 'bs4', 'dtale',
        'lime', 'matplotlib', 'shap', 'sweetviz', 'aix360', 'explainerdashboard', 'pandas_profiling',
        'pandas_visual_analysis', 'markupsafe==2.0.1'
    ],

)
