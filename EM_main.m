max_itr = 100;
tol = 1e-6;
thet_0 = [1;1;3;2];
d = length(thet_0);

sig=3*norm(thet_0);
p=5;
N=10^p;

ys = generate_observations(thet_0, N, sig, 'Gaussian');
thet_init = thet_0+[-0.3;0;0.1;0.2];

[theta_ks, num_itr] = runEM(thet_init, ys, max_itr, sig, tol);

%%%%post processing


thet_errs = zeros(1, num_itr);
for i = 1:num_itr
    thet = theta_ks(:,i);
    thet_errs(i) = relative_error(thet_0, thet);
end
idx = 1:num_itr;
s = strcat('sigma=',num2str(sig),', num samples = 10^',num2str(p), ', d= ',num2str(d));
figure;plot(idx, thet_errs(idx)); title(s);ylabel('theta rel error');

