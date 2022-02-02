clear all;
close all; 
clc;
%% Data Loading
% 파일 선택 대화상자 열기(확장자는 dat,  경로는 Select a ECG file로 지정)
[filename, pathname] = uigetfile('*.dat', 'Select a ECG file');

% 파일 정보 저장 (파일 열기)
fid = fopen(filename);

% 샘플링 주파수 250,섀년 샘플링 정리 2fmax<fs,이득(증폭값) 200,zero-value 값 0
fs = 250;                                                                   
fmax = fs/2;                                                                
gain = 200;                                                                 
zeroval = 0; 

% 1시간 데이터 읽기  
data_load_time = fs*3600;

% 데이터 저장 (3 x data_load_time 부호없는 8비트 정수형 배열) 
file_read = fread(fid,[3, data_load_time], 'uint8'); 

% 파일 닫기
fclose(fid); 

%% 데이터 이진화 (컴퓨터가 음수와 양수를 구별하기 위해 2의 보수를 이용)
% 지정된 개수의 위치만큼 이동, 비트별 논리 AND연산
Ch2_H = bitshift(file_read(2 , : ),-4);                                   
Ch2_L = bitand(file_read(2 , : ),15);                                       

% 부호비트 추출
sign_Ch2_H = bitshift(bitand(file_read(2, :), 128), 5);
sign_Ch2_L = bitshift(bitand(file_read(2, :), 8), 9);

% 부호비트를 빼준다
Ch_ECG = (bitshift(Ch2_L,8) + file_read(1,:) - sign_Ch2_L - zeroval)/gain;
Ch_PPG = (bitshift(Ch2_H,8) + file_read(3,:) - sign_Ch2_H - zeroval)/gain;

% 스크린 사이즈를 설정하고 f1으로 식별된 객체의 속성 위치에 대한 값을 지정한다
screen_size = get(0,'ScreenSize');
f1=figure(1);
f2=figure(2);
f3=figure(3);
f4=figure(4);
set(f1,'Position',[0 0 screen_size(3) screen_size(4)]);
set(f2,'Position',[0 0 screen_size(3) screen_size(4)]);
set(f3,'Position',[0 0 screen_size(3) screen_size(4)]);
set(f4,'Position',[0 0 screen_size(3) screen_size(4)]);

% 데이터 구간 설정 (35.1분~35.5분 동안 데이터 분석)
t_start = 35.1 * 60 * fs;
t_end = 35.5 * 60 * fs;
ECG = Ch_ECG(t_start : t_end);
t = [35.1 : 1/(fs*60) : 35.5];
figure(1);
subplot(6,1,1);
plot(t,ECG,'b');
title('ECG');
xlabel('second[sec]');ylabel('Voltage[uv]');

%% High pass filter
% HPF(5Hz)
wn = 5/fmax;

% 버터워스필터는 차단 주파수 wn을 갖는 n차 고역통과필터
[B,A] = butter(3,wn,'high');

% ECG 신호 영위상 필터링
Hp_ECG = filtfilt(B,A,ECG);
subplot(6,1,2);
plot(t,Hp_ECG,'b');
title('HPF 5Hz');
xlabel('second[sec]');ylabel('Voltage[uv]');

%% R peak detection
% Low pass filtering
A_L = [1 -2 1];
B_L = [1 0 0 0 0 0 -2 0 0 0 0 0 1];
R_Lp_ECG = filter(B_L,A_L,Hp_ECG);
subplot(6,1,3);
plot(t,R_Lp_ECG,'b');
axis([min(t) max(t) -50  50]); 
title('LPF');
xlabel('second[sec]');ylabel('Voltage[uv]');

% High pass filtering
A_H = [1 -1];
B_H = zeros(1,33);B_H(1) = -1/32;B_H(17) = 1;B_H(18) = -1;B_H(33) = 1/32;
R_Hp_ECG = filter(B_H,A_H,R_Lp_ECG);
subplot(6,1,4);
plot(t,R_Hp_ECG,'b');
axis([min(t) max(t) -50  50]); 
title('HPF');
xlabel('second[sec]');ylabel('Voltage[uv]');

% Derivative
A_D = 1;
B_D = [1/4, 1/8, 0, -1/8, -1/4];
R_D_ECG = filter(B_D,A_D,R_Hp_ECG);
subplot(6,1,5);
plot(t,R_D_ECG,'b');
axis([min(t) max(t) -20  20]); 
title('Derivative');
xlabel('second[sec]');ylabel('Voltage[uv]');

% Squaring
R_S_ECG = R_D_ECG.^2;
subplot(6,1,6);
plot(t,R_S_ECG,'b');
axis([min(t) max(t) 0 400]); 
title('Squaring');
xlabel('second[sec]');ylabel('Voltage[uv]');

%% R peak detection w/ Pan & Tompkins 
figure(2);
% 250샘플링값에 80msec(20sample; 윈도우 길이)이동 평균
% movmean(입력배열, 윈도우길이)
ECG_mov = movmean(R_S_ECG,20);   

% 플롯 기준선은 Squaring된 ECG의 최대값 * 0.2
ECG_MAX = max(ECG_mov)*0.2;                                                   
subplot(4,1,1);
plot(t,ECG_mov,'b');

% 플롯에 기준선 추가하기 (기준선 'red')
ECG_hline = refline([0 ECG_MAX]);                                           
ECG_hline.Color = 'r';                                                      
title('moving average');
xlabel('second[sec]');ylabel('Voltage[uv]');

% 신호간의 지연 추정 (원신호 ECG 기본 값에서 moving average 까지 지연 추정)
ECG_del = finddelay(ECG,ECG_mov);

% 24만큼 왼쪽으로 지연
ECG_d_left = circshift(ECG_mov,-ECG_del);

% 기준선 위의 값이 임계값
ECG_th = ECG_d_left > ECG_MAX;                                                
subplot(4,1,2);
plot(t,ECG_th,'b');
title('threshold');
xlabel('second[sec]');ylabel('Voltage[uv]');

% ECG_임계값 표시하기
subplot(4,1,3);
hold on;
plot(t,Hp_ECG,'b',t, ECG_th,'r');                                           
title('ECG threshold');
xlabel('second[sec]');ylabel('Voltage[uv]');

% R 피크 표시하기 ( [국소최댓값, 피크위치]=findpeak(입력값) )
subplot(4,1,4);
tmp_fun = ECG_th .* Hp_ECG;
[ECG_R_pks,ECG_R_loc] = findpeaks(tmp_fun,'MinPeakHeight',1);
plot(t,ECG,'b',t(ECG_R_loc),ECG(ECG_R_loc),'ro');
title('Rpeak');
xlabel('second[sec]');ylabel('Voltage[uv]');
 
%% iso-electric
% R피크 기준으로 108ms 이전값을 20ms씩 표준편차 최소값을 구함 
% window length 20ms (끝점 위치를 만들기 위해)
wnd_20ms = 20*fs/1000;

% 108ms 이전 (시작점 위치를 만들기 위해)
Sel_108ms = 108*fs/1000;

% for 반복문을 통해 인덱스 설정 (1부터 ECG R피크위치의 길이)
for i=1:length(ECG_R_loc)                                                   
    ECG_search_start_p = ECG_R_loc(i) - Sel_108ms;                          
    ECG_search_end_p = ECG_R_loc(i)-wnd_20ms;
    
    % tmp 값 임시 저장 (저장하고 사용되면 이전값은 없어진다)
    tmp_fun = ECG(ECG_search_start_p : ECG_search_end_p);
    
    % movstd(이동표준편차) 입력배열과 윈도우길이를 표준편차
    M = movstd(tmp_fun,wnd_20ms);
    
    % 최소 인덱스는 최소 이동표준편차다.                                   
    [~, min_idx] = min(M);
    
    % 시작점에서 최소 인덱스를 더하여 iso_x의 위치를 구한다.
    Iso_x(i) = min_idx + ECG_search_start_p;                                
    hold on;
    scatter(t(Iso_x(i)),ECG(Iso_x(i)),'r*');
end
     legend('filtered ECG','R peak','iso-electric point')

%% interpolation with spline
figure(3);
% x축구간 설정
x_re = 1 : length(ECG); 

% iso_x로 선형 보간 interp1(샘플 점, 샘플 값, 쿼리 점)
ECG_lin = interp1(Iso_x, ECG(Iso_x),x_re,'linear');                        
ECG_dif_li = ECG - ECG_lin;                                               
subplot(2,2,1);
plot(t,ECG,'b',t(ECG_R_loc),ECG(ECG_R_loc),'ro');
title('Rpeak');
xlabel('second[sec]');ylabel('Voltage[uv]');
hold on;
plot(t,ECG_lin,'r');

% iso_x로 3차 보간 interp1(샘플 점, 샘플 값, 쿼리 점)
subplot(2,2,2);
ECG_cub = interp1(Iso_x, ECG(Iso_x),x_re,'pchip'); 
ECG_dif_cub = ECG - ECG_cub;
plot(t,ECG,'b',t(ECG_R_loc),ECG(ECG_R_loc),'ro');
title('Rpeak');
xlabel('second[sec]');ylabel('Voltage[uv]');
hold on;
plot(t,ECG_cub,'r');

subplot(2,2,3);
plot(t,ECG_dif_li,'b');
title('linear diff');
xlabel('second[sec]');ylabel('Voltage[uv]');

subplot(2,2,4);
plot(t,ECG_dif_cub,'b');
title('cubic diff');
xlabel('second[sec]');ylabel('Voltage[uv]');

    
%% QRST peak detection (cubic interpolation)
figure(4)
% cubic 보간에서 R피크 점 찾기 [국소최댓값, 피크위치]=findpeak(입력값)
subplot(2,1,1);
[ECG_R_pks,ECG_R_loc] = findpeaks(ECG_dif_cub, 'MinPeakHeight',1.7,...
    'MinPeakDistance',0.1);
plot(t,ECG_dif_cub);
hold on;
scatter(t(ECG_R_loc),ECG_dif_cub(ECG_R_loc),'ro');

% Q wave
% for 반복문을 통해 인덱스 설정 (1부터 ECG R피크위치의 길이)
for i = 1:length(ECG_R_loc)
    Q_start_p(i) = ECG_R_loc(i)-10;
    Q_end_p(i) = ECG_R_loc(i);
    
    % tmp 값 임시 저장 (저장하고 사용되면 이전값은 없어진다)
    Q_tmp = ECG_dif_cub(Q_start_p(i) : Q_end_p(i));
    [~, min_idx] = min(Q_tmp);  
    idx = min_idx + Q_start_p;
    
    % S wave
    S_start_p(i) = ECG_R_loc(i);
    S_end_p(i) = ECG_R_loc(i) + 7;
    
    % tmp 값 임시 저장 (저장하고 사용되면 이전값은 없어진다)
    S_tmp = ECG_dif_cub(S_start_p : S_end_p);
    [~, min_idx] = min(S_tmp);  
    idx_2 = min_idx + S_start_p;
 
end
scatter(t(idx),ECG_dif_cub(idx),'r*');
scatter(t(idx_2),ECG_dif_cub(idx_2),'rs');

% T wave
% cubic 보간에서 T피크 점 찾기 [국소최댓값, 피크위치]=findpeak(입력값)
[~,ECG_T_loc] = findpeaks(ECG_dif_cub,'MinPeakDistance',50);

% T wave y축 구간 설정 (0.2부터 1까지)
locs_Twave = ECG_T_loc(ECG_dif_cub(ECG_T_loc)> 0.2 & ...
    ECG_dif_cub(ECG_T_loc) < 1);
hold on;
plot(t(locs_Twave),ECG_dif_cub(locs_Twave),'bx');

% for 반복문을 통해 인덱스 설정 (1부터 ECG T파형의 길이)
% T peak에서 좌우로 시작점과 끝점을 그림
for i = 1:length(locs_Twave)
    T_start_p(i) = locs_Twave(i)-30;
    T_end_p(i) = locs_Twave(i);
    T_tmp = ECG_dif_cub(T_start_p : T_end_p);
    [~, min_idx] = min(T_tmp);
    idx = min_idx + T_start_p;
    
    % T 파형 그리기
    T_start_p_2(i) = locs_Twave(i);
    T_end_p_2(i) = locs_Twave(i)+20;
    T_tmp_2 = ECG_dif_cub(T_start_p_2 : T_end_p_2);
    [~, min_idx] = min(T_tmp_2);  
    idx_2 = min_idx + T_start_p_2;
    scatter(t(idx_2(i)),ECG_dif_cub(idx_2(i)),'b<');
    hold on;
    plot(t(idx(i):idx_2(i)), ECG_dif_cub(idx(i):idx_2(i)),'g');
end

legend('filtered ECG','R peak','Q peak','S peak','T wave','T end')
title('ECG QRST Point');
xlabel('t');ylabel('amplitude');
