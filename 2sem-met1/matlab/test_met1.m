close all
clear all
clc
format long

%%
student = 0; % here is your number

for n_dim = [2, 3]
    N = student;
    A = [
        4,  1,      1
        1,  6+.2*N, -1
        1,  -1,     8+.2*N
        ];
    b = [1, -2, 3]';
    x0 = [0, 0, 0]';

    A = A(1:n_dim, 1:n_dim);
    b = b(1:n_dim);
    x0 = x0(1:n_dim);
    
    eps = 1e-6;

    methods = {'met1_mngs', 'met1_mps'};

    styles = {'mo-', 'b.:'};

    figure(n_dim)
    title(['���������� ��� ����������� ' num2str(n_dim)])
    hold on
    for i = 1:numel(methods)
        method = methods{i};
        disp(['running ', method, ' test for student #', num2str(student)])
        [X, Y] = feval(method, A, b, x0, eps);

        x1 = linsolve(A, -b);
        y1 = 1/2*x1'*A*x1 + b'*x1;

        assert(all(x0 == X(:,1)))
        assert(norm(x1 - X(:, end)) < 1e-3)
        assert(norm(y1 - Y(:, end)) < eps)
        plot(-log10(Y - y1), styles{i})
    end
    xlabel('����� ��������')
    ylabel('��������')
    legend show
end
