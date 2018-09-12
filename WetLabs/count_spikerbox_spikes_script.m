%% script:  count_spikerbox_spikes_script 
%% Jennifer M. Groh, Center for Cognitive Neuroscience, Duke University, November 2011
%%
%% This script will count spikes based on a user-specified voltage
%% threshold and time window.  
%%
%% Assumes you have loaded a wav file using File>Import Data and the
%% variable "data" has the timeseries, and "fs" has the sampling rate in Hz


timebase=(1:length(data(:,1)))/fs;  %% The times of each sample.

figure(1);
clf;
plot(timebase,data,'b-');
xlabel('Time (seconds)');
ylabel('Voltage (arbitrary units)');

%threshold=.8;
%starttime=1.7;
%stoptime=2.5;

%% Next steps allow user to choose some simple parameters

starttime=input('Start counting spikes when? ');
stoptime=input('Stop counting spikes when? ');
threshold=input('Spike is anything above what threshold value? ');

%% Limit data to time period of interest
usedata=data(timebase>starttime & timebase<stoptime);
usetimebase=timebase(timebase>starttime & timebase<stoptime);

%% Find voltage values above threshold and corresponding times within that time period
spike_voltages=usedata(usedata>threshold);
spike_times=usetimebase(usedata>threshold);

%% Now we need to count each spike once even though the voltage will likely stay
%% above threshold for more than a sample

diff_spike_times=conv(spike_times,[1 -1]);  %% differentiates the time values, i.e. values here are the 
                                            %% time intervals between successive values of
                                            %% "spike_times".
diff_spike_times=diff_spike_times(2:end-1); %% trim the first and last value
spike_gaps=diff_spike_times>1/1000;  %% Separate individual spikes must be at least 1/1000 of a second apart


hold on;
plot(spike_times(spike_gaps),spike_voltages(spike_gaps),'g.');  % plot a green dot on each spike we count
disp('There were')
sum(spike_gaps)
disp('during your spike counting window.');




