from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

setup(
    name='ai-sense',
    packages=find_packages(exclude=['examples']),
    version='0.1.0',
    license='MIT',
    description='AI Sensing model in Wireless and mobile platform ',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Xingwei Wang',
    author_email='wxwjkl123@gmail.com',
    url='https://github.com/xibrer/ai-sense',
    keywords=[
        'artificial intelligence',
        '1d',
    ],
    install_requires=[
        'einops>=0.7.0',
        'torch>=1.10',
    ],
    setup_requires=[
        'pytest-runner',
    ],
    tests_require=[
        'pytest',
        'torch>=1.12.1',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
)
