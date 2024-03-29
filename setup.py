import setuptools


setuptools.setup(
    name='ancinf',
    version='0.0.1',
    author='Dmitry Glyzin, Alexey Shmelev, Kenenbek Arzumanov',
    author_email='dglyzin@hse.ru',
    description='',
    zip_safe=False,
    python_requires='>=3.9',
    packages=setuptools.find_packages(),
    include_package_data=True,
    #install_requires=list(open('requirements.txt').read().split()),
    #entry_points={
    #    'console_scripts': [
    #        'simulate=pf3dmodel.simulate:main',
    #        'showdata=pf3dmodel.showdata:main',
    #    ]
    #},
)
