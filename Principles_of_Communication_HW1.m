%%
% Name:         Bithiah Ngan
% Course:       ELC 4350 
% Assignment:   HW 1 (Chapter 7 Digital Filtering and the DFT)
% Date:         3/8/2021

%% Exercise 7.5
f =100; Ts=1/1000; time=10.0; % freq , sampling interval , time
t=Ts : Ts : time ; % define a time vector
w=sin(2* pi * f * t ); % define the sinusoid
N=2^10; % si z e o f a n al y si s window
ssf =(-N/ 2:N/2 -1)/( Ts*N); % frequency vector
fw=fft(w(1:N) ); % do DFT/FFT
fws =fftshift( fw ); % shift it for plotting 
plot ( ssf , abs( fws )) % plot magnitude spectrum
title( 'Sine Wave via the FFT ( Different Frequencies) ')
xlabel('Frequency')
ylabel('Magnitude')
hold on 
f =2200; Ts=1/1000; time=10.0; % freq , sampling interval , time
t=Ts : Ts : time ; % define a time vector
w=sin(2* pi * f * t ); % define the sinusoid
N=2^10; % si z e o f a n al y si s window
ssf =(-N/ 2:N/2 -1)/( Ts*N); % frequency vector
fw=fft(w(1:N) ); % do DFT/FFT
fws =fftshift( fw ); % shift it for plotting
plot ( ssf , abs( fws )) % plot magnitude spectrum
legend('Frequency 200 Hz', ',Frequency 2200 Hz')
hold off
  
fprintf('Exercise 7.5 \n')
fprintf('(a): As frequency increases the sampling intervel increases. \n')
clear
f =100; Ts=1/500; time=10.0; % freq , sampling interval , time
t=Ts : Ts : time ; % define a time vector
w=sin(2* pi * f * t ); % define the sinusoid
N=2^10; % si z e o f a n al y si s window
ssf =(-N/ 2:N/2 -1)/( Ts*N); % frequency vector
fw=fft(w(1:N) ); % do DFT/FFT
fws =fftshift( fw ); % shift it for plotting
plot ( ssf , abs( fws )) % plot magnitude spectrum
title( 'Sine Wave via the FFT ( Different Ts) ')
xlabel('Frequency')
ylabel('Magnitude')
hold on 

f =100; Ts=1/50; time=10.0; % freq , sampling interval , time
t=Ts : Ts/9 : time ; % define a time vector
w=sin(2* pi * f * t ); % define the sinusoid
N=2^10; % si z e o f a n al y si s window
ssf =(-N/ 2:N/2 -1)/( Ts*N); % frequency vector
fw=fft(w(1:N) ); % do DFT/FFT
fws =fftshift( fw ); % shift it for plotting
plot ( ssf , abs( fws )) % plot magnitude spectrum
legend('Ts = 1/500' ,'Ts = 1/50')
hold off 

fprintf('(b): As ts increases the frequency decreases. \n')
fprintf('But the location of the peak does not change.\n') 

f =100; Ts=1/1000; time=10.0; % freq , sampling interval , time
t=Ts : Ts : time ; % define a time vector
w=sin(2* pi * f * t ); % define the sinusoid
N=2^4; % si z e o f a n al y si s window
ssf =(-N/ 2:N/2 -1)/( Ts*N); % frequency vector
fw=fft(w(1:N) ); % do DFT/FFT
fws =fftshift( fw ); % shift it for plotting
plot ( ssf , abs( fws )) % plot magnitude spectrum
hold on
title( 'Sine Wave via the FFT ( Different Size) ')
xlabel('Frequency')
ylabel('Magnitude')
f =100; Ts=1/1000; time=1000; % freq , sampling interval , time
t=Ts : Ts : time ; % define a time vector
w=sin(2* pi * f * t ); % define the sinusoid
N=2^5; % si z e o f a n al y si s window
ssf =(-N/ 2:N/2 -1)/( Ts*N); % frequency vector
fw=fft(w(1:N) ); % do DFT/FFT
fws =fftshift( fw ); % shift it for plotting
plot ( ssf , abs( fws )) % plot magnitude spectrum
legend('N = 2^4', ' N = 2^5')
hold off
fprintf('(c): As N becomes too large,the curve is just one line.\n')
fprintf('     As N becomes too small,the curve is flat. \n')
fprintf('But the location of the peak does not change.\n') 
%% Exercise 7.6
f =100; Ts=1/1000; time=10.0; % freq , sampling interval , time
t=Ts : Ts : time ; % define a time vector
w=sin(3* pi * f * t ).^2; % define the sinusoid
N=2^10; % si z e o f a n al y si s window
ssf =(-N/ 2:N/2 -1)/( Ts*N); % frequency vector
fw=fft(w(1:N) ); % do DFT/FFT
fws =fftshift( fw ); % shift it for plotting
plot ( ssf , abs( fws )) % plot magnitude spectrum
title( 'Sin Wave via the FFT ')
xlabel('Frequency')
ylabel('Magnitude')
hold on 
w=sin(3* pi * f * t ).^3; % define the sinusoid
N=2^10; % si z e o f a n al y si s window
ssf =(-N/ 2:N/2 -1)/( Ts*N); % frequency vector
fw=fft(w(1:N) ); % do DFT/FFT
fws =fftshift( fw ); % shift it for plotting
plot ( ssf , abs( fws )) % plot magnitude spectrum
hold on
w=sin(3* pi * f * t ).^10; % define the sinusoid
N=2^10; % si z e o f a n al y si s window
ssf =(-N/ 2:N/2 -1)/( Ts*N); % frequency vector
fw=fft(w(1:N) ); % do DFT/FFT
fws =fftshift( fw ); % shift it for plotting
plot ( ssf , abs( fws )) % plot magnitude spectrum
hold off
legend('Sin^2','Sin^3', 'Sin^4^0')
fprintf('Exercise 7.6 \n')
fprintf('The spectrum of sin^2 is at -300Hz,0 Hz and 300 Hz.\n')
fprintf('The spectrum of sin^3 is at -450Hz,-150 Hz, 150 Hz and 450 Hz.\n')
fprintf('After sin^10, the spectrum frequency does not change,\n')
fprintf('But the magnitude of signal increase as the power of sin increases.\n')

%% Exercise 7.7
f =100; Ts=1/1000; time=10.0; % freq , sampling interval , time
t=Ts : Ts : time ; % define a time vector
w=sinc(3* pi * f * t ); % define the sinusoid
N=2^10; % si z e o f a n al y si s window
ssf =(-N/ 2:N/2 -1)/( Ts*N); % frequency vector
fw=fft(w(1:N) ); % do DFT/FFT
fws =fftshift( fw ); % shift it for plotting

plot ( ssf , abs( fws )) % plot magnitude spectrum
title( 'Sinc Wave via the FFT ')
xlabel('Frequency')
ylabel('Magnitude')

w=sinc(3* pi * f * t ).^2; % define the sinusoid
N=2^10; % si z e o f a n al y si s window
ssf =(-N/ 2:N/2 -1)/( Ts*N); % frequency vector
fw=fft(w(1:N) ); % do DFT/FFT
fws =fftshift( fw ); % shift it for plotting

plot ( ssf , abs( fws )) % plot magnitude spectrum
title( 'Sinc^2 Wave via the FFT ')
xlabel('Frequency')
ylabel('Magnitude')
fprintf('Exercise 7.7 \n')

%% Exercise 7.8

f =100; Ts=1/1000; time=10.0; % freq , sampling interval , time
t=Ts : Ts : time ; % define a time vector
j = sqrt(-1);
w=sin(t)+j*exp(-t); % define the sinusoid
N=2^10; % si z e o f a n al y si s window
ssf =(-N/ 2:N/2 -1)/( Ts*N); % frequency vector
fw=fft(w(1:N) ); % do DFT/FFT
fws =fftshift( fw ); % shift it for plotting
plot ( ssf , abs( fws )) % plot magnitude spectrum
title( 'sin(t)+j*exp(-t)Wave via the FFT ')
xlabel('Frequency')
ylabel('Magnitude')
fprintf('Exercise 7.8 \n')

%% Exercise 7.9
%Let w=sin(2*pi*f*t+phi). For phi = 0, 0.2, 0.4, 0.8, 1.5, 3.14, find the phase of
%the FFT output at the frequencies ±f.
f =100; Ts=1/1000; time=10.0; % freq , sampling interval , time
t=Ts : Ts : time ; % define a time vector
phi = 0;
w=sin(2* pi * f * t + phi ); % define the sinusoid
N=2^10; % si z e o f a n al y si s window
ssf =(-N/ 2:N/2 -1)/( Ts*N); % frequency vector
fw=fft(w(1:N) ); % do DFT/FFT
fws =fftshift( fw ); % shift it for plotting 
plot ( ssf , angle( fws )) % plot magnitude spectrum
title( 'Sine Wave via the FFT ( Different Frequencies) ')
xlabel('Frequency')
ylabel('Magnitude')
hold on 
phi = 0.2;
w=sin(2* pi * f * t + phi ); % define the sinusoid
N=2^10; % si z e o f a n al y si s window
ssf =(-N/ 2:N/2 -1)/( Ts*N); % frequency vector
fw=fft(w(1:N) ); % do DFT/FFT
fws =fftshift( fw ); % shift it for plotting 
plot ( ssf , angle( fws )) % plot magnitude spectrum
hold on 
phi = 0.4;
w=sin(2* pi * f * t + phi ); % define the sinusoid
N=2^10; % si z e o f a n al y si s window
ssf =(-N/ 2:N/2 -1)/( Ts*N); % frequency vector
fw=fft(w(1:N) ); % do DFT/FFT
fws =fftshift( fw ); % shift it for plotting 
plot ( ssf , angle( fws )) % plot magnitude spectrum
hold on 
phi = 0.8;
w=sin(2* pi * f * t + phi ); % define the sinusoid
N=2^10; % si z e o f a n al y si s window
ssf =(-N/ 2:N/2 -1)/( Ts*N); % frequency vector
fw=fft(w(1:N) ); % do DFT/FFT
fws =fftshift( fw ); % shift it for plotting 
plot ( ssf , angle( fws )) % plot magnitude spectrum
hold on 
phi = 1.5;
w=sin(2* pi * f * t + phi ); % define the sinusoid
N=2^10; % si z e o f a n al y si s window
ssf =(-N/ 2:N/2 -1)/( Ts*N); % frequency vector
fw=fft(w(1:N) ); % do DFT/FFT
fws =fftshift( fw ); % shift it for plotting 
plot ( ssf , angle( fws )) % plot magnitude spectrum
hold on 
phi = 3.14;
w=sin(2* pi * f * t + phi ); % define the sinusoid
N=2^10; % si z e o f a n al y si s window
ssf =(-N/ 2:N/2 -1)/( Ts*N); % frequency vector
fw=fft(w(1:N) ); % do DFT/FFT
fws =fftshift( fw ); % shift it for plotting 
plot ( ssf , angle( fws )) % plot magnitude spectrum
hold off 
legend('Phi = 0','Phi = 0.2','Phi = 0.4', 'Phi = 0.8','Phi = 1.5','Phi = 3.14')

%% Exercise 7.10
filename= 'gong.wav' ; % name o f wave f i l e
[ x , sr]=audioread( filename ); % read in wavefile
Ts=1/sr ; % sample interval & # of samples
N=2^(14); x= x( 1 :N )'; % length for analysis
sound(x , 1/ Ts ) % play sound ( i f possible )
time=Ts * (0: length(x) - 1); % time base for plotting
 plot ( time , x) % and pl o t top fi g u re
xlabel('Time in seconds')
ylabel('Amplitude')
title('Time and frequency plots of the gong waveform')
magx= abs( fft(x)); % t a ke FFT magnitude
ssf =(0:N/2 - 1)/(Ts*N); % freq base for plotting

plot ( ssf , magx ( 1 :N/ 2 )/1000 ) % p l o t mag spectrum
xlabel('Frequency in Hertz')
ylabel('Magnitude')
% The total time is T = NTs, where N is the number of samples in the analysis
%the spectrum of the gong sound during the first 0.1 s
fprintf('Exercise 7.10 \n')
fprintf('Frequency does not change when N changes,\n')
fprintf('Frequency of FFT is between 500 to 550 Hz\n')
fprintf('Magnitude is smaller as time decreases as the noise goes\n')
%% Exercise 7.11
fprintf('Exercise 7.11 \n')
filename= 'gong.wav' ; % name o f wave f i l e
[ x , sr]=audioread( filename ); % read in wavefile
Ts=1/sr ; % sample interval & # of samples
N=2^(16); x= x( 1 :N )'; % length for analysis
sound(x , 1/ Ts ) % play sound ( i f possible )
time=Ts * (0: length(x) - 1); % time base for plotting
 plot ( time , x) % and pl o t top fi g u re
xlabel('Time in seconds')
ylabel('Amplitude')
title('Time and frequency plots of the gong waveform')
magx= abs( fft(x)); % t a ke FFT magnitude
ssf =(0:N/2 - 1)/(Ts*N); % freq base for plotting

semilogy( ssf , magx ( 1 :N/ 2 )/1000 ) % p l o t mag spectrum
grid on
xlabel('Frequency in Hertz')
ylabel('Magnitude')
fprintf('The graph shows that it is continous and there are spikes.\n')
fprintf('But 500-550 Hz is peak frequency\n')
%% Exercise 7.12
fprintf('Exercise 7.12 \n')
filename= 'gong2.wav' ; % name o f wave f i l e
[ x , sr]=audioread( filename ); % read in wavefile
Ts=1/sr ; % sample interval & # of samples
N=2^(16); x= x( 1 :N )'; % length for analysis
sound(x , 1/ Ts ) % play sound ( i f possible )
time=Ts * (0: length(x) - 1); % time base for plotting
plot ( time , x) % and pl o t top fi g u re
xlabel('Time in seconds')
ylabel('Amplitude')
title('Time and frequency plots of the gong2 waveform ( N = 2^1^6)')

N=2^(14); x= x( 1 :N )'; % length for analysis
sound(x , 1/ Ts ) % play sound ( i f possible )
time=Ts * (0: length(x) - 1); % time base for plotting
plot ( time , x) % and pl o t top fi g u re
xlabel('Time in seconds')
ylabel('Amplitude')
title('Time and frequency plots of the gong2 waveform ( N = 2^1^4)')
N=2^(12); x= x( 1 :N )'; % length for analysis
sound(x , 1/ Ts ) % play sound ( i f possible )
time=Ts * (0: length(x) - 1); % time base for plotting
plot ( time , x) % and pl o t top fi g u re
xlabel('Time in seconds')
ylabel('Amplitude')
title('Time and frequency plots of the gong2 waveform (N = 2^1^2)')
fprintf('At a variety of windows with different N and tiems,\n')
fprintf('The peak frequency does not change but as N smaller there is less intensity.\n')

%% Exercise 7.13
filename= 'suspence.wav' ; % name o f wave f i l e
[ x , sr]=audioread( filename ); % read in wavefile
Ts=1/sr ; % sample interval & # of samples
N=2^(18); x= x( 1 :N )'; % length for analysis
sound(x , 1/ Ts ) % play sound ( i f possible )
time=Ts * (0: length(x) - 1); % time base for plotting
subplot(2,1,1);plot ( time , x) % and pl o t top fi g u re
xlabel('Time in seconds')
ylabel('Amplitude')
title('Time and frequency plots of the Suspence waveform')
magx= abs( fft(x)); % t a ke FFT magnitude
ssf =(0:N/2 - 1)/(Ts*N); % freq base for plotting

subplot(2,1,2);plot ( ssf , magx ( 1 :N/ 2 )/1000 ) % p l o t mag spectrum
xlabel('Frequency in Hertz')
ylabel('Magnitude')

% The total time is T = NTs, where N is the number of samples in the analysis
%the spectrum of the gong sound during the first 0.1 s
fprintf('Exercise 7.13 \n')


%% Exercise 7.15

% waystofiltIIR.m implementing IIR filters

a=[1 -0.8]; lena=length(a)-1;       % autoregressive coefficients
b=[2]; lenb=length(b);              % moving average coefficients
d=randn(1,20);                      % data to filter
if lena>=lenb                       % dimpulse requires lena>=lenb
  h=impz(b,a);                      % impulse response of filter
  yfilt=filter(h,1,d);               % filter x[k] with h[k]
end
yfilt2=filter(b,a,d);               % filter directly using a and b
y=zeros(lena,1); x=zeros(lenb,1);   % initial states in filter
for k=1:length(d)-lenb              % time domain method
  x=[d(k);x(1:lenb-1)];             % past values of inputs
  ytim(k)=-a(2:lena+1)*y+b*x;       % directly calculate y[k]
  y=[ytim(k);y(1:lena-1)];          % past values of outputs
end
fprintf('Exercise 7.15 \n')
