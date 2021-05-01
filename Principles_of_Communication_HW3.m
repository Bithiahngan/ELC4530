%%
% Name:         Bithiah Ngan
% Course:       ELC 4350 
% Assignment:   HW 3 11.3, 11.4, 12.1, 12.2, 12.3, 13.1, 13.2
% Date:         4/30/2021

%% Problem 1 
%the data sequence is drawn from the alphabet ±1, ±3, ±5

% eyediag.m plot eye diagrams for pulse shape ps
N=1000; m=pam(N,6,1);            % random signal of length N
M=20; mup=zeros(1,N*M);          % oversampling factor of M
mup(1:M:N*M)=m;                  % oversample by M
ps=hamming(M);                   % hamming pulse of width M
x=filter(ps,1,mup);              % convolve pulse shape
neye=5;                          % size of groups
c=floor(length(x)/(neye*M));     % number of eyes to plot
xp=x(N*M-neye*M*c+1:N*M);        % ignore transients at start
figure(1)
plot(reshape(xp,neye*M,c))       % plot in groups
figure(1)
title('Eye diagram for hamming pulse shape')

figure(2)                      % used to plot figure eyediag3
N=1000; m=pam(N,6,1);          % random +/-1 signal of length N
M=20; mup=zeros(1,N*M); mup(1:M:N*M)=m;  % oversampling by factor of M
ps=ones(1,M);                            % square pulse width M
x=filter(ps,1,mup);            % convolve pulse shape with mup
neye=5;
c=floor(length(x)/(neye*M))
xp=x(N*M-neye*M*c+1:N*M);      % dont plot transients at start
q=reshape(xp,neye*M,c);        % plot in clusters of size 5*Mt=(1:198)/50+1;
subplot(3,1,1), plot(q)
title('Eye diagram for rectangular pulse shape')

N=1000; m=pam(N,6,1);          % random +/-1 signal of length N
M=20; mup=zeros(1,N*M); mup(1:M:N*M)=m;  % oversampling by factor of M
ps=hamming(M);                           % square pulse width M
x=filter(ps,1,mup);            % convolve pulse shape with mup
%x=x+0.15*randn(size(x));
neye=5;
c=floor(length(x)/(neye*M))
xp=x(N*M-neye*M*c+1:N*M);      % dont plot transients at start
q=reshape(xp,neye*M,c);        % plot in clusters of size 5*Mt=(1:198)/50+1;
subplot(3,1,2), plot(q)
title('Eye diagram for hamming pulse shape')

N=1000; m=pam(N,6,1);          % random +/-1 signal of length N
M=20; mup=zeros(1,N*M); mup(1:M:N*M)=m;  % oversampling by factor of M
L=10; ps=srrc(L,0,M,50);
ps=ps/max(ps);         % sinc pulse shape L symbols wide
x=filter(ps,1,mup);    % convolve pulse shape with mup
%x=x+0.15*randn(size(x));
neye=5;
c=floor(length(x)/(neye*M))
xp=x(N*M-neye*M*(c-3)+1:N*M);  % dont plot transients at start
q=reshape(xp,neye*M,c-3);      % plot in clusters of size 5*Mt=(1:198)/50+1;
subplot(3,1,3), plot(q)
axis([0,100,-3,3])
title('Eye diagram for sinc pulse shape')

figure(3)
N=1000; m=pam(N,6,5);          % random +/-1 signal of length N
M=20; mup=zeros(1,N*M); mup(1:M:N*M)=m;  % oversampling by factor of M
ps=hamming(M);                           % square pulse width M
x=filter(ps,1,mup);    % convolve pulse shape with mup
%x=x+0.15*randn(size(x));
neye=5;
c=floor(length(x)/(neye*M))
xp=x(N*M-neye*M*c+1:N*M);   % dont plot transients at start
q=reshape(xp,neye*M,c);     % plot in clusters of size 5*Mt=(1:198)/50+1;
t=(1:neye*M)/M;
subplot(4,1,1), plot(t,q)
hold on
title('Eye diagram for the T-wide hamming pulse shape')

N=1000; m=pam(N,6,5);                    % random signal of length N
M=20; mup=zeros(1,N*M); mup(1:M:N*M)=m;  % oversampling by factor of M
ps=hamming(2*M);                         % hamming pulse of width M
x=filter(ps,1,mup);                      % convolve pulse shape with mup
neye=5; c=floor(length(x)/(neye*M));     % number of eyes to plot
xp=x(N*M-neye*M*(c-3)+1:N*M);            % dont plot transients at start
t=(1:neye*M)/M;
subplot(4,1,2),plot(t,reshape(xp,neye*M,c-3))
title('Eye diagram for the 2T-wide hamming pulse shape')

N=1000; m=pam(N,4,5);                    % random signal of length N
M=20; mup=zeros(1,N*M); mup(1:M:N*M)=m;  % oversampling by factor of M
ps=hamming(3*M);                         % hamming pulse of width M
x=filter(ps,1,mup);                      % convolve pulse shape with mup
neye=5; c=floor(length(x)/(neye*M));     % number of eyes to plot
xp=x(N*M-neye*M*(c-3)+1:N*M);            % dont plot transients at start
subplot(4,1,3),plot(t,3/4*reshape(xp,neye*M,c-3))
title('Eye diagram for the 3T-wide hamming pulse shape')

N=1000; m=pam(N,6,5);                    % random signal of length N
M=20; mup=zeros(1,N*M); mup(1:M:N*M)=m;  % oversampling by factor of M
ps=hamming(5*M);                         % hamming pulse of width M
x=filter(ps,1,mup);                      % convolve pulse shape with mup
neye=5; c=floor(length(x)/(neye*M));     % number of eyes to plot
xp=x(N*M-neye*M*(c-3)+1:N*M);            % dont plot transients at start
subplot(4,1,4),plot(t,3/5*reshape(xp,neye*M,c-3))
title('Eye diagram for the 5T-wide hamming pulse shape')
xlabel('symbols')
hold off

L=10; ps=srrc(L,0,M,0);                 % sinc pulse shape L symbols wide
ps=ones(1,M);                           % square pulse width M

%% Question 2 
%to add noise to the pulse-shaped signal x
% eyediag.m plot eye diagrams for pulse shape ps
v = v*randn;
figure(2)                   % used to plot figure eyediag3
N=1000; m=pam(N,2,1);          % random +/-1 signal of length N
M=20; mup=zeros(1,N*M); mup(1:M:end)=m; % oversampling by factor of M
ps=ones(1,M);                           % square pulse width M
x=filter(ps,1,mup)+ x*v;   % convolve pulse shape with mup
neye=5;
c=floor(length(x)/(neye*M));
xp=x(end-neye*M*c+1:end);                       % dont plot transients at start
q=reshape(xp,neye*M,c);     % plot in clusters of size 5*Mt=(1:198)/50+1;
subplot(3,1,1), plot(q)
title('Eye diagram for rectangula pulse shape')

N=1000; m=pam(N,2,1);          % random +/-1 signal of length N
M=20; mup=zeros(1,N*M); mup(1:M:end)=m; % oversampling by factor of M
ps=hamming(M);                           % square pulse width M
x=filter(ps,1,mup)+ x*v;    % convolve pulse shape with mup
%x=x+0.15*randn(size(x));
neye=5;
c=floor(length(x)/(neye*M));
xp=x(end-neye*M*c+1:end);                       % dont plot transients at start
q=reshape(xp,neye*M,c);     % plot in clusters of size 5*Mt=(1:198)/50+1;
subplot(3,1,2), plot(q)
title('Eye diagram for hamming pulse shape')

N=1000; m=pam(N,2,1);          % random +/-1 signal of length N
M=20; mup=zeros(1,N*M); mup(1:M:end)=m; % oversampling by factor of M
L=10; ps=srrc(L,0,M,50); 
ps=ps/max(ps); % sinc pulse shape L symbols wide
x=filter(ps,1,mup) + x*v;    % convolve pulse shape with mup
%x=x+0.15*randn(size(x));
neye=5;
c=floor(length(x)/(neye*M));
xp=x(end-neye*M*(c-3)+1:end);       % dont plot transients at start
q=reshape(xp,neye*M,c-3);     % plot in clusters of size 5*Mt=(1:198)/50+1;
subplot(3,1,3), plot(q)
axis([0,100,-3,3])
title('Eye diagram for sinc pulse shape')
hold off
%% Question 3 (12.1)
%(a) How does mu affect the convergence rate? What range of stepsizes works?
% clockrecDD.m: clock recovery minimizing 4-PAM cluster variance
% to minimize J(tau) = (Q(x(kT/M+tau))-x(kT/M+tau))^2

% prepare transmitted signal
n=10000;                         % number of data points
m=2;                             % oversampling factor
beta=0.3;                        % rolloff parameter for srrc
l=50;                            % 1/2 length of pulse shape (in symbols)
chan=[1];                        % T/m "channel"
toffset=-0.3;                    % initial timing offset
pulshap=srrc(l,beta,m,toffset);  % srrc pulse shape with timing offset
s=pam(n,2,5);                    % random data sequence with var=5
sup=zeros(1,n*m);                % upsample the data by placing...
sup(1:m:n*m)=s;                  % ... m-1 zeros between each data point
hh=conv(pulshap,chan);           % ... and pulse shape
r=conv(hh,sup);                  % ... to get received signal
matchfilt=srrc(l,beta,m,0);      % matched filter = srrc pulse shape
x=conv(r,matchfilt);             % convolve signal with matched filter
% clock recovery algorithm
tnow=l*m+1; tau=0; xs=zeros(1,n);   % initialize variables
tausave=zeros(1,n); tausave(1)=tau; i=0;
mu=0.01;                            % algorithm stepsize
delta=0.1;                          % time for derivative
while tnow<length(x)-2*l*m          % run iteration
  i=i+1;
  xs(i)=interpsinc(x,tnow+tau,l);   % interp value at tnow+tau
  x_deltap=interpsinc(x,tnow+tau+delta,l); % value to right
  x_deltam=interpsinc(x,tnow+tau-delta,l); % value to left
  dx=x_deltap-x_deltam;             % numerical derivative
  qx=quantalph(xs(i),[-1,1]);  % quantize to alphabet
  tau=tau+mu*dx*(qx-xs(i));         % alg update: DD
  tnow=tnow+m; tausave(i)=tau;      % save for plotting
end

% plot results
subplot(2,1,1), plot(xs(1:i-2),'b.')        % plot constellation diagram
title('constellation diagram');
ylabel('estimated symbol values')
subplot(2,1,2), plot(tausave(1:i-2))        % plot trajectory of tau
ylabel('offset estimates'), xlabel('iterations')
hold off
%% Question 4 (12.2)
% prepare transmitted signal
n=10000;                         % number of data points
m = 2;                             % oversampling factor
beta=0.3;                        % rolloff parameter for srrc
l=50;                            % 1/2 length of pulse shape (in symbols)
chan=[1];                        % T/m "channel"
toffset=-0.3;                    % initial timing offset
pulshap=ones(1,M);  % srrc pulse shape with timing offset
s=pam(n,4,5);                    % random data sequence with var=5
sup=zeros(1,n*m);                % upsample the data by placing...
sup(1:m:n*m)=s;                  % ... m-1 zeros between each data point
hh=conv(pulshap,chan);           % ... and pulse shape
r=conv(hh,sup);                  % ... to get received signal
matchfilt=ones(1,M);      % matched filter = srrc pulse shape
x=conv(r,matchfilt);      % convolve signal with matched filter
% clock recovery algorithm
tnow=l*m+1; tau=0; xs=zeros(1,n);   % initialize variables
tausave=zeros(1,n); tausave(1)=tau; i=0;
mu=0.01;                            % algorithm stepsize
delta=0.1;                          % time for derivative
while tnow<length(x)-2*l*m          % run iteration
  i=i+1;
  xs(i)=interpsinc(x,tnow+tau,l);   % interp value at tnow+tau
  x_deltap=interpsinc(x,tnow+tau+delta,l); % value to right
  x_deltam=interpsinc(x,tnow+tau-delta,l); % value to left
  dx=x_deltap-x_deltam;             % numerical derivative
  qx=quantalph(xs(i),[3,-1,1,3]);  % quantize to alphabet
  tau=tau+mu*dx*(qx-xs(i));         % alg update: DD
  tnow=tnow+m; tausave(i)=tau;      % save for plotting
end

% plot results
subplot(2,1,1), plot(xs(1:i-2),'b.')        % plot constellation diagram
title('constellation diagram');
ylabel('estimated symbol values')
subplot(2,1,2), plot(tausave(1:i-2))        % plot trajectory of tau
ylabel('offset estimates'), xlabel('iterations')
hold off

%% Question 5 (12.3)

% prepare transmitted signal
n=10000;                         % number of data points
m = 2;                             % oversampling factor
beta=0.3;                        % rolloff parameter for srrc
l=50;                            % 1/2 length of pulse shape (in symbols)
chan=[1];                        % T/m "channel"
toffset=-0.3;                    % initial timing offset
pulshap=ones(1,M);  % srrc pulse shape with timing offset
s=pam(n,4,5);                    % random data sequence with var=5
sup=zeros(1,n*m);                % upsample the data by placing...
sup(1:m:n*m)=s;                  % ... m-1 zeros between each data point
hh=conv(pulshap,chan);           % ... and pulse shape
r=conv(hh,sup);                  % ... to get received signal
matchfilt=srrc(l,beta,m,0);      % matched filter = srrc pulse shape
x=conv(r,matchfilt)+ randn(size(x));      % convolve signal with matched filter
% clock recovery algorithm
tnow=l*m+1; tau=0; xs=zeros(1,n);   % initialize variables
tausave=zeros(1,n); tausave(1)=tau; i=0;
mu=0.01;                            % algorithm stepsize
delta=0.1;                          % time for derivative
while tnow<length(x)-2*l*m          % run iteration
  i=i+1;
  xs(i)=interpsinc(x,tnow+tau,l);   % interp value at tnow+tau
  x_deltap=interpsinc(x,tnow+tau+delta,l); % value to right
  x_deltam=interpsinc(x,tnow+tau-delta,l); % value to left
  dx=x_deltap-x_deltam;             % numerical derivative
  qx=quantalph(xs(i),[3,-1,1,3]);  % quantize to alphabet
  tau=tau+mu*dx*(qx-xs(i));         % alg update: DD
  tnow=tnow+m; tausave(i)=tau;      % save for plotting
end

% plot results
subplot(2,1,1), plot(xs(1:i-2),'b.')        % plot constellation diagram
title('constellation diagram');
ylabel('estimated symbol values')
subplot(2,1,2), plot(tausave(1:i-2))        % plot trajectory of tau
ylabel('offset estimates'), xlabel('iterations')
hold off

%% Question 6

%LSequalizer.m find a LS equalizer f for the channel b
b=[ 0.5 1 -.6]                  % define channel
sd = 0.2
m=1000; s=sign(randn(1,m));       % binary source of length m
r=filter(b,1,s)+sd*randn(size(s));           % output of channel
n=3;                              % length of equalizer - 1
delta=2;                          % use delay <=n*length(b) 
p=length(r)-delta;
R=toeplitz(r(n+1:p),r(n+1:-1:1)); % build matrix R 
S=s(n+1-delta:p-delta)';          % and vector S
f=inv(R'*R)*R'*S;                 % calculate equalizer f
Jmin=S'*S-S'*R*inv(R'*R)*R'*S   % Jmin for this f and delta
y=filter(f,1,r);             % equalizer is a filter
dec=sign(y);                      % quantize and find errors
err=0.5*sum(abs(dec(delta+1:end)-s(1:end-delta)))







%% functions 
% seq=pam(len,M,Var);
% Create an M-PAM source sequence with
% length 'len'  and variance 'Var'
function seq=pam(len,M,Var)
seq=(2*floor(M*rand(1,len))-M+1)*sqrt(3*Var/(M^2-1));
end


function s=srrc(syms, beta, P, t_off)

% s=srrc(syms, beta, P, t_off);
% Generate a Square-Root Raised Cosine Pulse
%      'syms' is 1/2 the length of srrc pulse in symbol durations
%      'beta' is the rolloff factor: beta=0 gives the sinc function
%      'P' is the oversampling factor
%      't_off' is the phase (or timing) offset

if nargin==3, t_off=0; end                 % if unspecified, offset is 0
k=-syms*P+1e-8+t_off:syms*P+1e-8+t_off;           % sampling indices as a multiple of T/P
if (beta==0), beta=1e-8; end;                     % numerical problems if beta=0
s=4*beta/sqrt(P)*(cos((1+beta)*pi*k/P)+...        % calculation of srrc pulse
  sin((1-beta)*pi*k/P)./(4*beta*k/P))./...
  (pi*(1-16*(beta*k/P).^2));
end
function y=interpsinc(x, t, l, beta)
% y=interpsinc(x, t, l, beta)
% interpolate to find a single point using the direct method
%        x = sampled data
%        t = place at which value desired
%        l = one sided length of data to interpolate
%        beta = rolloff factor for srrc function
%             = 0 is a sinc
if nargin==3, beta=0; end           % if unspecified, beta is 0
tnow=round(t);                       % create indices tnow=integer part
tau=t-round(t);                      % plus tau=fractional part
s_tau=srrc(l,beta,1,tau);            % interpolating sinc at offset tau
x_tau=conv(x(tnow-l:tnow+l),s_tau);  % interpolate the signal
y=x_tau(2*l+1);                      % the new sample

end

% y=quantalph(x,alphabet)
%
% quantize the input signal x to the alphabet
% using nearest neighbor method
% input x - vector to be quantized
%       alphabet - vector of discrete values that y can take on
%                  sorted in ascending order
% output y - quantized vector
function y=quantalph(x,alphabet)
alphabet=alphabet(:);
x=x(:);
alpha=alphabet(:,ones(size(x)))';
dist=(x(:,ones(size(alphabet)))-alpha).^2;
[v,i]=min(dist,[],2);
y=alphabet(i);
end