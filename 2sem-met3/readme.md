���������: http://www.apmath.spbu.ru/ru/structure/depts/is/task6-2013.pdf

�� �������� ����� ��� ����� ������� ������ ������������� ����� �������� �����������.

* `matlab/`
  * `test_met3.m` - ������, ������� ��������� ���� ���������� �������������.
  * `met3_func.m` - ������� ������� (�������� �����). ������ ���� ���������������: ��������� �� ���� �������� ������ � ���������� �������� ������.
  * `met3_approximate.m` - ���� ���������� �������������. ����� ������� ������ ��������� ��� ������������� (*).
  * `LegendrePoly.m` - ������� ��� ������� ������������� ���������� �������� (��� ���������� ����� ���������).

* `py/`
  * `test_met3.py` - ����, ������� ��������� ���� ���������� �������������.
  * `met3.py` - ��� ������ ���� ���� func(x) � approx(X0, Y0, X1, approx_type, dim).

---
(*) ��������� ���� �������������:
* 0 - �� ��������� x^k;
* 1 - �� ��������� ����������� ��������;
* 2 - �����������: ������ �������� ������ ��� `f(-1) = f(1)`, ��� ������ �� ���������, ������ ����������� ����������� � ������ ������� / �������� ����������. 
��� ��������� ������ �����: https://ru.wikipedia.org/wiki/������������������_���_����� 