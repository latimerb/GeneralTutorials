function[num_spikes]=count_spikerbox_spikes(data,fs,threshold,starttime,stoptime);
%%function[num_spikes]=count_spikerbox_spikes(data,fs,threshold,starttime,stoptime);
%%
%% Jennifer M. Groh, Center for Cognitive Neuroscience, Duke University, November 2011
%%
%% This function will count spikes based on a user-specified voltage
%% threshold and time window (in seconds).  Import your wav file using File>Import data
%% and then pass the resulting variables "data" (which has the timeseries) and "fs"
%% (which has the sampling rate in Hz) as arguments.
%%


timebase=(1:length(data(:,1)))/fs;  %% The times of each sample.

figure(1);                  % Makes figure 1 the active figure
clf;                        % clf means "clear figure" (in case there was already something on it)
plot(timebase,data,'b-');   % Syntax is plot(x,y, and a string describing how you want the 
                            %      data to look).  Here, it is b for blue
                            %      and - for a line.  Type "help plot" at the command prompt for
                            %      more options.
                            
xlabel('Time (seconds)');
ylabel('Voltage (arbitrary units)');

%% Limit data to time period of interest
usedata=data(timebase>starttime & timebase<stoptime);  
        %% selects values of the variable data for which the 
        %% corresponding values of timebase lie in the desired range.
usetimebase=timebase(timebase>starttime & timebase<stoptime);

%% Find voltage values above threshold and corresponding times within that time period
spike_voltages=usedata(usedata>threshold);
spike_times=usetimebase(usedata>threshold);

%% Now we need to count each spike once even though the voltage will likely stay
%% above threshold for more than a sample

diff_spike_times=conv(spike_times,[1 -1]);  %% differentiates the time values, i.e. values here are the 
                                            %% time intervals between successive values of
                                            %% "spike_times". (Conv is
                                            %% convolution).
diff_spike_times=diff_spike_times(2:end-1); %% trim the first and last value
spike_gaps=diff_spike_times>1/1000;  %% Separate individual spikes must be at least 1/1000 of a second apart


hold on;  %% superimpose the next plot command on the existing graph
plot(spike_times(spike_gaps),spike_voltages(spike_gaps),'g.');  % plot a green dot on each spike we count
disp('There were');         
num_spikes=sum(spike_gaps)  %no semicolon here!  So the result of this statement
                            % will be displayed at the command prompt
disp('during your spike counting window.');




