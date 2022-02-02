clear all;
close all; 
clc;
%% Data Loading
% ���� ���� ��ȭ���� ����(Ȯ���ڴ� dat,  ��δ� Select a ECG file�� ����)
[filename, pathname] = uigetfile('*.dat', 'Select a ECG file');

% ���� ���� ���� (���� ����)
fid = fopen(filename);

% ���ø� ���ļ� 250,���� ���ø� ���� 2fmax<fs,�̵�(������) 200,zero-value �� 0
fs = 250;                                                                   
fmax = fs/2;                                                                
gain = 200;                                                                 
zeroval = 0; 

% 1�ð� ������ �б�  
data_load_time = fs*3600;

% ������ ���� (3 x data_load_time ��ȣ���� 8��Ʈ ������ �迭) 
file_read = fread(fid,[3, data_load_time], 'uint8'); 

% ���� �ݱ�
fclose(fid); 

%% ������ ����ȭ (��ǻ�Ͱ� ������ ����� �����ϱ� ���� 2�� ������ �̿�)
% ������ ������ ��ġ��ŭ �̵�, ��Ʈ�� �� AND����
Ch2_H = bitshift(file_read(2 , : ),-4);                                   
Ch2_L = bitand(file_read(2 , : ),15);                                       

% ��ȣ��Ʈ ����
sign_Ch2_H = bitshift(bitand(file_read(2, :), 128), 5);
sign_Ch2_L = bitshift(bitand(file_read(2, :), 8), 9);

% ��ȣ��Ʈ�� ���ش�
Ch_ECG = (bitshift(Ch2_L,8) + file_read(1,:) - sign_Ch2_L - zeroval)/gain;
Ch_PPG = (bitshift(Ch2_H,8) + file_read(3,:) - sign_Ch2_H - zeroval)/gain;

% ��ũ�� ����� �����ϰ� f1���� �ĺ��� ��ü�� �Ӽ� ��ġ�� ���� ���� �����Ѵ�
screen_size = get(0,'ScreenSize');
f1=figure(1);
f2=figure(2);
f3=figure(3);
f4=figure(4);
set(f1,'Position',[0 0 screen_size(3) screen_size(4)]);
set(f2,'Position',[0 0 screen_size(3) screen_size(4)]);
set(f3,'Position',[0 0 screen_size(3) screen_size(4)]);
set(f4,'Position',[0 0 screen_size(3) screen_size(4)]);

% ������ ���� ���� (35.1��~35.5�� ���� ������ �м�)
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

% ���Ϳ������ʹ� ���� ���ļ� wn�� ���� n�� ���������
[B,A] = butter(3,wn,'high');

% ECG ��ȣ ������ ���͸�
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
% 250���ø����� 80msec(20sample; ������ ����)�̵� ���
% movmean(�Է¹迭, ���������)
ECG_mov = movmean(R_S_ECG,20);   

% �÷� ���ؼ��� Squaring�� ECG�� �ִ밪 * 0.2
ECG_MAX = max(ECG_mov)*0.2;                                                   
subplot(4,1,1);
plot(t,ECG_mov,'b');

% �÷Կ� ���ؼ� �߰��ϱ� (���ؼ� 'red')
ECG_hline = refline([0 ECG_MAX]);                                           
ECG_hline.Color = 'r';                                                      
title('moving average');
xlabel('second[sec]');ylabel('Voltage[uv]');

% ��ȣ���� ���� ���� (����ȣ ECG �⺻ ������ moving average ���� ���� ����)
ECG_del = finddelay(ECG,ECG_mov);

% 24��ŭ �������� ����
ECG_d_left = circshift(ECG_mov,-ECG_del);

% ���ؼ� ���� ���� �Ӱ谪
ECG_th = ECG_d_left > ECG_MAX;                                                
subplot(4,1,2);
plot(t,ECG_th,'b');
title('threshold');
xlabel('second[sec]');ylabel('Voltage[uv]');

% ECG_�Ӱ谪 ǥ���ϱ�
subplot(4,1,3);
hold on;
plot(t,Hp_ECG,'b',t, ECG_th,'r');                                           
title('ECG threshold');
xlabel('second[sec]');ylabel('Voltage[uv]');

% R ��ũ ǥ���ϱ� ( [�����ִ�, ��ũ��ġ]=findpeak(�Է°�) )
subplot(4,1,4);
tmp_fun = ECG_th .* Hp_ECG;
[ECG_R_pks,ECG_R_loc] = findpeaks(tmp_fun,'MinPeakHeight',1);
plot(t,ECG,'b',t(ECG_R_loc),ECG(ECG_R_loc),'ro');
title('Rpeak');
xlabel('second[sec]');ylabel('Voltage[uv]');
 
%% iso-electric
% R��ũ �������� 108ms �������� 20ms�� ǥ������ �ּҰ��� ���� 
% window length 20ms (���� ��ġ�� ����� ����)
wnd_20ms = 20*fs/1000;

% 108ms ���� (������ ��ġ�� ����� ����)
Sel_108ms = 108*fs/1000;

% for �ݺ����� ���� �ε��� ���� (1���� ECG R��ũ��ġ�� ����)
for i=1:length(ECG_R_loc)                                                   
    ECG_search_start_p = ECG_R_loc(i) - Sel_108ms;                          
    ECG_search_end_p = ECG_R_loc(i)-wnd_20ms;
    
    % tmp �� �ӽ� ���� (�����ϰ� ���Ǹ� �������� ��������)
    tmp_fun = ECG(ECG_search_start_p : ECG_search_end_p);
    
    % movstd(�̵�ǥ������) �Է¹迭�� ��������̸� ǥ������
    M = movstd(tmp_fun,wnd_20ms);
    
    % �ּ� �ε����� �ּ� �̵�ǥ��������.                                   
    [~, min_idx] = min(M);
    
    % ���������� �ּ� �ε����� ���Ͽ� iso_x�� ��ġ�� ���Ѵ�.
    Iso_x(i) = min_idx + ECG_search_start_p;                                
    hold on;
    scatter(t(Iso_x(i)),ECG(Iso_x(i)),'r*');
end
     legend('filtered ECG','R peak','iso-electric point')

%% interpolation with spline
figure(3);
% x�౸�� ����
x_re = 1 : length(ECG); 

% iso_x�� ���� ���� interp1(���� ��, ���� ��, ���� ��)
ECG_lin = interp1(Iso_x, ECG(Iso_x),x_re,'linear');                        
ECG_dif_li = ECG - ECG_lin;                                               
subplot(2,2,1);
plot(t,ECG,'b',t(ECG_R_loc),ECG(ECG_R_loc),'ro');
title('Rpeak');
xlabel('second[sec]');ylabel('Voltage[uv]');
hold on;
plot(t,ECG_lin,'r');

% iso_x�� 3�� ���� interp1(���� ��, ���� ��, ���� ��)
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
% cubic �������� R��ũ �� ã�� [�����ִ�, ��ũ��ġ]=findpeak(�Է°�)
subplot(2,1,1);
[ECG_R_pks,ECG_R_loc] = findpeaks(ECG_dif_cub, 'MinPeakHeight',1.7,...
    'MinPeakDistance',0.1);
plot(t,ECG_dif_cub);
hold on;
scatter(t(ECG_R_loc),ECG_dif_cub(ECG_R_loc),'ro');

% Q wave
% for �ݺ����� ���� �ε��� ���� (1���� ECG R��ũ��ġ�� ����)
for i = 1:length(ECG_R_loc)
    Q_start_p(i) = ECG_R_loc(i)-10;
    Q_end_p(i) = ECG_R_loc(i);
    
    % tmp �� �ӽ� ���� (�����ϰ� ���Ǹ� �������� ��������)
    Q_tmp = ECG_dif_cub(Q_start_p(i) : Q_end_p(i));
    [~, min_idx] = min(Q_tmp);  
    idx = min_idx + Q_start_p;
    
    % S wave
    S_start_p(i) = ECG_R_loc(i);
    S_end_p(i) = ECG_R_loc(i) + 7;
    
    % tmp �� �ӽ� ���� (�����ϰ� ���Ǹ� �������� ��������)
    S_tmp = ECG_dif_cub(S_start_p : S_end_p);
    [~, min_idx] = min(S_tmp);  
    idx_2 = min_idx + S_start_p;
 
end
scatter(t(idx),ECG_dif_cub(idx),'r*');
scatter(t(idx_2),ECG_dif_cub(idx_2),'rs');

% T wave
% cubic �������� T��ũ �� ã�� [�����ִ�, ��ũ��ġ]=findpeak(�Է°�)
[~,ECG_T_loc] = findpeaks(ECG_dif_cub,'MinPeakDistance',50);

% T wave y�� ���� ���� (0.2���� 1����)
locs_Twave = ECG_T_loc(ECG_dif_cub(ECG_T_loc)> 0.2 & ...
    ECG_dif_cub(ECG_T_loc) < 1);
hold on;
plot(t(locs_Twave),ECG_dif_cub(locs_Twave),'bx');

% for �ݺ����� ���� �ε��� ���� (1���� ECG T������ ����)
% T peak���� �¿�� �������� ������ �׸�
for i = 1:length(locs_Twave)
    T_start_p(i) = locs_Twave(i)-30;
    T_end_p(i) = locs_Twave(i);
    T_tmp = ECG_dif_cub(T_start_p : T_end_p);
    [~, min_idx] = min(T_tmp);
    idx = min_idx + T_start_p;
    
    % T ���� �׸���
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
