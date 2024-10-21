from setuptools import setup

package_name = 'traj_opt'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lab',
    maintainer_email='lab@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'test_script = traj_opt.test_script:main',
            'cf_publisher = traj_opt.cf_publisher:main',
            'mpc_solver = traj_opt.mpc_solver:main'
        ],
    },
)
