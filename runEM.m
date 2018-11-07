function [theta_ks, num_itr] = runEM(thet_init, X, max_itr, sigma, tol)
% Expectation maximization algorithm for multireference alignment.
% X: data (each column is an observation)
% sigma: noise standard deviation affecting measurements
% thet_init: initial guess for the signal 
% tol: EM stops iterating if two subsequent iterations are closer than tol
%      in 2-norm, up to circular shift (default: 1e-5).

%
% May 2017
% https://arxiv.org/abs/1705.00641
% https://github.com/NicolasBoumal/MRA

 % X contains N observations, each of length d
[d, N] = size(X);
fftX = fft(X);  
sqnormX = repmat(sum(abs(X).^2, 1), d, 1);
theta_ks = zeros(d, max_itr);
theta_ks(:,1)=thet_init;
num_itr = 1;
fftx = fft(thet_init);

    for iter = 2 : max_itr
       % disp(iter)
        fftx_new = EM_iteration(fftx, fftX, sqnormX, sigma);
        re = relative_error(ifft(fftx), ifft(fftx_new));
       % disp(re);
        if re < tol
            theta_ks(:, num_itr+1:max_itr) = [];
            break;
        end
        num_itr = num_itr + 1;
        theta_ks(:,iter) = ifft(fftx_new);      
        fftx = fftx_new;

    end

end

% Execute one iteration of EM with current estimate of the DFT of the
% signal given by fftx, and DFT's of the observations stored in fftX, and
% squared 2-norms of the observations stored in sqnormX, and noise level
% sigma.
function fftx_new = EM_iteration(fftx, fftX, sqnormX, sigma)

    C = ifft(bsxfun(@times, conj(fftx), fftX));
    T = (2*C - sqnormX)/(2*sigma^2);
    T = bsxfun(@minus, T, max(T, [], 1));
    W = exp(T);
    W = bsxfun(@times, W, 1./sum(W, 1));
    fftx_new = mean(conj(fft(W)).*fftX, 2);

end