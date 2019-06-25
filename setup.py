# encoding =utf-8

from os import path
import codecs
from setuptools import setup, find_packages

setup(
    name='bert_encoder',
    version='0.1.1',
    description='convert text to vector via pretrained bert model',
    url='https://github.com/4AI/bert-encoder',
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='4AI',
    author_email='xmlee97@gmail.com',
    license='MIT',
    packages=find_packages(),
    zip_safe=False,
    extras_require={
        'cpu': ['tensorflow>=1.10.0<=1.13.0'],
        'gpu': ['tensorflow-gpu>=1.10.0<=1.13.0'],
    },
    classifiers=(
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence :: Natural Language Processing',
    ),
    keywords='bert nlp recognition tensorflow machine learning sentence encoding embedding serving',
)
