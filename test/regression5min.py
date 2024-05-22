import sys
from workdir.repeat_test_10.repeat_test_10 import repeat_test_10

test_names = ['arrakis10', 'arrakis100', 'arrakis10-100', 'arrakis10-100-100']
lst_error = []

for i in range(len(test_names)):
    if repeat_test_10(test_names[i]) == 1:
        lst_error.append(test_names[i])
print('Тест не прошли: ', lst_error)




