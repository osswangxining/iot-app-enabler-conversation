from setuptools import setup

__version__ = None  # Avoids IDE errors, but actual version is read from version.py
exec (open('mynlu/version.py').read())

tests_requires = [
    "pytest-pep8",
    "pytest-services",
    "pytest-cov",
    "pytest-twisted",
    "treq"
]

install_requires = [
    "pathlib",
    "cloudpickle",
    "gevent",
    "klein",
    "boto3",
    "typing",
    "future",
    "six",
    "tqdm",
    "requests",
    "jsonschema",
    "matplotlib"
]

extras_requires = {
    'test': tests_requires,
    'spacy': ["sklearn", "scipy", "numpy"],
    'mitie': ["mitie", "numpy"],
}

setup(
    name='mynlu',
    packages=[
        'mynlu',
        'mynlu.classifiers',
        'mynlu.cli',
        'mynlu.config',
        'mynlu.extractors',
        'mynlu.featurizers',
        'mynlu.model',
        'mynlu.nlp',
        'mynlu.parsers',
        'mynlu.pipeline',
        'mynlu.tokenizers',
        'mynlu.trainers',
        'mynlu.utils'
    ],
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6"
    ],
    version=__version__,
    install_requires=install_requires,
    tests_require=tests_requires,
    extras_require=extras_requires,
    include_package_data=True,
    description="NLU a natural language parser for bots",
    author='',
    author_email='',
    url="",
    keywords=["NLP", "bots"],
    download_url=""
)
