from setuptools import setup, find_packages

setup(
    name='Senti-AM-LSTM',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'openpyxl',  # For reading/writing Excel files
        'tensorflow',
        'scikit-learn',
        'vaderSentiment',
        'textblob',
        'numpy',
        'joblib',
    ],
    entry_points={
        'console_scripts': [
            'senti-am-lstm=senti_am_lstm_tool.predict_price:main',  # CLI entry point
        ],
    },
    include_package_data=True,
    package_data={
        '': ['models/*.h5', 'models/*.pkl'],  # Ensure model files are included in the package
    },
    author='Praveenkumar A', 'Girish Kumar Jha' and 'Sharanbasappa D Madival'.
    author_email='praveenkumarupm@gmail.com',
    description='SENTI-AM-LSTM: A novel sentiment-enhanced attention-based LSTM model for agricultural futures price prediction'.,
    url='https://github.com/your-repo/senti-am-lstm-tool',  # Link to the project repository
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9.13',
)
