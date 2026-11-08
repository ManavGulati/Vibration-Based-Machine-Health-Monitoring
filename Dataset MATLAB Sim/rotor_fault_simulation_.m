% Faults:
%   healthy        : residual rotor unbalance only
%   unbalance      : increased eccentricity -> 1X rotating force
%   misalignment   : additional calibrated 1X and 2X coupling forces
%   bearing_outer  : repeated BPFO impact pulses exciting a housing mode
%
% IMPORTANT ENGINEERING LIMITATION:
% This is a physics-informed SYNTHETIC teaching / ML-pipeline model. The
% coefficients for the rotor, coupling, bearing impacts, environment and
% sensor are illustrative. They must be calibrated against a real machine
% and real sensor measurements before engineering or diagnostic claims are
% made. A simulation is not a substitute for experimental validation.

clear; clc; close all;
rng(17, 'twister');                 % reproducible dataset generation

%% 1. Simulation settings
cfg.fs = 12800;                     % sample rate [Hz]
cfg.duration_s = 1.0;               % saved record length [s]
cfg.warmup_s = 0.30;                % discard startup transient [s]
cfg.rpmList = [1200 1800 2400];     % nominal commanded operating speeds
cfg.loadLevels = [0.30 0.50 0.75 1.00]; % fraction of rated load
cfg.nRepeats = 5;                   % independent records per operating point
cfg.randomSeed = 17;

% Operating-condition variability (illustrative starting values).
cfg.speedRippleFraction = 0.0035;  % slow periodic speed ripple (fraction)
cfg.speedRandomFraction = 0.0015;  % correlated speed variation (fraction)
cfg.speedRippleHzRange = [0.20 0.80];
cfg.loadRippleFraction = 0.035;    % absolute load-fraction ripple
cfg.loadRandomFraction = 0.018;    % correlated load variation
cfg.loadStepProbability = 0.20;    % chance of one load step per record
cfg.loadStepFractionRange = [0.08 0.18];
cfg.loadDisturbanceForce_N = 0.025;

% Thermal environment. Temperature is recorded for every simulated run.
cfg.ambientTemperatureRange_C = [18 32];
cfg.temperatureRiseAtRatedLoad_C = 40;
% These small thermal coefficients are modelling assumptions, not universal
% material properties. Set to zero if temperature should be metadata only.
cfg.stiffnessTempCoeff_perC = -1.0e-4;
cfg.dampingTempCoeff_perC = 1.0e-3;

%% 2. Rotor lateral model (illustrative parameters)
cfg.rotorMass_kg = 5.0;
cfg.rotorNaturalFrequency_Hz = 75;
cfg.rotorDampingRatio = 0.03;
cfg.machineStiffnessScatter = 0.015; % run-to-run setup variation
cfg.machineDampingScatter = 0.08;

% Residual unbalance exists even in a healthy machine.
cfg.unbalanceMass_kg = 0.05;
cfg.residualEccentricity_m = 10e-6;
% Additional eccentricity by severity index 0,1,2,3 [m].
cfg.faultEccentricity_m = [0 50e-6 100e-6 150e-6];
cfg.unbalanceAssemblyGainRange = [0.90 1.10];

%% 3. Misalignment force model
% EFFECTIVE coupling-force amplitudes, not universal conversions from
% shaft offset / angle. Calibrate against measured coupling forces.
cfg.misalignment1X_N = [0 0.08 0.16 0.24];
cfg.misalignment2X_N = [0 0.12 0.24 0.36];
cfg.misalignmentGainRange = [0.90 1.10];

%% 4. Bearing geometry and local housing resonance
cfg.nRollingElements = 8;
cfg.ballDiameter_m = 8e-3;
cfg.pitchDiameter_m = 40e-3;
cfg.contactAngle_deg = 0;
cfg.impactSigma_s = 0.00015;        % Gaussian impact width [s]
cfg.bearingImpactPeak_N = [0 0.08 0.16 0.24];
cfg.bearingImpactGainRange = [0.75 1.25];

% Local bearing / housing resonance for impact ring-down.
cfg.housingModalMass_kg = 0.20;
cfg.housingResonance_Hz = 1500;
cfg.housingDampingRatio = 0.08;

%% 5. Background vibration and sensor model
cfg.externalTone1ForceRange_N = [0.003 0.018];
cfg.externalTone1FrequencyRange_Hz = [45 110];
cfg.externalTone2ForceRange_N = [0.001 0.010];
cfg.externalTone2FrequencyRange_Hz = [130 280];
cfg.externalLowFrequencyRange_Hz = [2 8];

cfg.sensorNoiseStdRange_m_s2 = [0.0005 0.0018];
cfg.coloredNoiseFraction = 0.35;
cfg.coloredNoiseAR = 0.96;
cfg.sensorBandwidthRange_Hz = [4500 6000];
cfg.sensorMountingGainRange = [0.90 1.10];
cfg.sensorDriftMax_m_s2 = 0.0006;
cfg.sensorResolution_m_s2 = 1e-5;
cfg.sensorRange_m_s2 = 50;

% Frequency-band half-width used to extract order-related peaks when the
% actual shaft speed fluctuates. At least two FFT bins are included.
cfg.orderBandFraction = 0.03;

%% 6. Dataset dimensions and preallocation
conditions = ["healthy", "unbalance", "misalignment", "bearing_outer"];
% Healthy severity 0; each fault class has severities 1, 2 and 3.
nRecords = numel(cfg.rpmList) * numel(cfg.loadLevels) * ...
    cfg.nRepeats * (1 + 3*3);
N = round(cfg.duration_s * cfg.fs);
t = (0:N-1) / cfg.fs;               % saved-record time axis
Nfull = round((cfg.duration_s + cfg.warmup_s) * cfg.fs);
tFull = (0:Nfull-1) / cfg.fs;       % includes discarded warm-up

% Backward-compatible main signal variable: measured X/radial channel.
signals = zeros(nRecords, N, 'single');
signalsY = zeros(nRecords, N, 'single');

metaRPM = zeros(nRecords, 1);       % nominal / commanded speed
metaCondition = strings(nRecords, 1);
metaSeverity = zeros(nRecords, 1);
metaReplicate = zeros(nRecords, 1);
metaRunID = zeros(nRecords, 1);
metaOperatingPointID = zeros(nRecords, 1);
metaNominalLoad = zeros(nRecords, 1);
metaMeanLoad = zeros(nRecords, 1);
metaRPMMean = zeros(nRecords, 1);
metaRPMStd = zeros(nRecords, 1);
metaTemperature = zeros(nRecords, 1);
metaAmbientTemperature = zeros(nRecords, 1);
metaBPFO = zeros(nRecords, 1);
metaNoiseStd = zeros(nRecords, 1);
metaSensorBandwidth = zeros(nRecords, 1);
metaMountingGain = zeros(nRecords, 1);
metaTransient = false(nRecords, 1);

featRMS = zeros(nRecords, 1);
featStd = zeros(nRecords, 1);
featPeak = zeros(nRecords, 1);
featKurtosis = zeros(nRecords, 1);
featCrest = zeros(nRecords, 1);
featAmp1X = zeros(nRecords, 1);
featAmp2X = zeros(nRecords, 1);
featAmpBPFO = zeros(nRecords, 1);
featRMS_Y = zeros(nRecords, 1);
featPeak_Y = zeros(nRecords, 1);
featKurtosis_Y = zeros(nRecords, 1);
featAmp1X_Y = zeros(nRecords, 1);
featAmp2X_Y = zeros(nRecords, 1);
featAmpBPFO_Y = zeros(nRecords, 1);

%% 7. Generate all operating conditions
record = 0;
for irpm = 1:numel(cfg.rpmList)
    rpmCommand = cfg.rpmList(irpm);

    for iload = 1:numel(cfg.loadLevels)
        nominalLoad = cfg.loadLevels(iload);
        operatingPointID = (irpm - 1)*numel(cfg.loadLevels) + iload;

        for ic = 1:numel(conditions)
            if conditions(ic) == "healthy"
                severityLevels = 0;
            else
                severityLevels = 1:3;
            end

            for severity = severityLevels
                for rep = 1:cfg.nRepeats
                    record = record + 1;

                    [accelX, accelY, info] = simulateOneRecord( ...
                        cfg, rpmCommand, conditions(ic), severity, ...
                        nominalLoad, tFull);
                    signals(record, :) = single(accelX);
                    signalsY(record, :) = single(accelY);

                    metaRPM(record) = rpmCommand;
                    metaCondition(record) = conditions(ic);
                    metaSeverity(record) = severity;
                    metaReplicate(record) = rep;
                    metaRunID(record) = record;
                    metaOperatingPointID(record) = operatingPointID;
                    metaNominalLoad(record) = nominalLoad;
                    metaMeanLoad(record) = info.meanLoad;
                    metaRPMMean(record) = info.meanRPM;
                    metaRPMStd(record) = info.stdRPM;
                    metaTemperature(record) = info.temperature_C;
                    metaAmbientTemperature(record) = info.ambientTemperature_C;
                    metaBPFO(record) = info.bpfoHz;
                    metaNoiseStd(record) = info.sensorNoiseStd;
                    metaSensorBandwidth(record) = info.sensorBandwidth_Hz;
                    metaMountingGain(record) = info.mountingGain;
                    metaTransient(record) = info.transientEvent;

                    % Feature extraction from measured X/radial acceleration.
                    x = accelX(:);
                    xCentered = x - mean(x);
                    rmsValue = sqrt(mean(x.^2));
                    variance2 = mean(xCentered.^2);
                    kurtValue = mean(xCentered.^4) / (variance2^2 + eps);

                    featRMS(record) = rmsValue;
                    featStd(record) = std(x);
                    featPeak(record) = max(abs(x));
                    featKurtosis(record) = kurtValue; % Pearson; Gaussian ~= 3
                    featCrest(record) = featPeak(record) / (rmsValue + eps);

                    [freqHz, amp] = oneSidedAmplitude(x, cfg.fs);
                    featAmp1X(record) = amplitudeInBand(freqHz, amp, ...
                        info.meanRPM/60, cfg.duration_s, cfg.orderBandFraction);
                    featAmp2X(record) = amplitudeInBand(freqHz, amp, ...
                        2*info.meanRPM/60, cfg.duration_s, cfg.orderBandFraction);
                    featAmpBPFO(record) = amplitudeInBand(freqHz, amp, ...
                        info.bpfoHz, cfg.duration_s, cfg.orderBandFraction);

                    % Selected Y-channel features provide directional context.
                    y = accelY(:);
                    yCentered = y - mean(y);
                    yVariance = mean(yCentered.^2);
                    featRMS_Y(record) = sqrt(mean(y.^2));
                    featPeak_Y(record) = max(abs(y));
                    featKurtosis_Y(record) = mean(yCentered.^4) / (yVariance^2 + eps);
                    [freqY, ampY] = oneSidedAmplitude(y, cfg.fs);
                    featAmp1X_Y(record) = amplitudeInBand(freqY, ampY, ...
                        info.meanRPM/60, cfg.duration_s, cfg.orderBandFraction);
                    featAmp2X_Y(record) = amplitudeInBand(freqY, ampY, ...
                        2*info.meanRPM/60, cfg.duration_s, cfg.orderBandFraction);
                    featAmpBPFO_Y(record) = amplitudeInBand(freqY, ampY, ...
                        info.bpfoHz, cfg.duration_s, cfg.orderBandFraction);
                end
            end
        end
    end
end

%% 8. Create tables and save files
metadata = table(metaRunID, metaOperatingPointID, metaRPM, metaRPMMean, ...
    metaRPMStd, metaNominalLoad, metaMeanLoad, metaTemperature, ...
    metaAmbientTemperature, metaCondition, metaSeverity, metaReplicate, ...
    metaBPFO, metaNoiseStd, metaSensorBandwidth, metaMountingGain, metaTransient, ...
    'VariableNames', {'Run_ID','OperatingPoint_ID','RPM_Command','RPM_ActualMean', ...
    'RPM_ActualStd','NominalLoad_Fraction','MeanLoad_Fraction','Temperature_C', ...
    'AmbientTemperature_C','Condition','Severity','Replicate','BPFO_Hz', ...
    'SensorNoiseStd_m_s2','SensorBandwidth_Hz','SensorMountingGain','TransientEvent'});

% Keep the original feature-column names and add operating-context and Y-axis
% features. RPM remains the nominal command for compatibility with old files.
features = table(metaRPM, metaCondition, metaSeverity, metaReplicate, metaBPFO, ...
    featRMS, featStd, featPeak, featKurtosis, featCrest, ...
    featAmp1X, featAmp2X, featAmpBPFO, ...
    metaRPMMean, metaRPMStd, metaNominalLoad*100, metaMeanLoad*100, ...
    metaTemperature, metaAmbientTemperature, metaNoiseStd, ...
    metaSensorBandwidth, metaMountingGain, metaTransient, metaRunID, ...
    metaOperatingPointID, featRMS_Y, featPeak_Y, featKurtosis_Y, ...
    featAmp1X_Y, featAmp2X_Y, featAmpBPFO_Y, ...
    'VariableNames', {'RPM','Condition','Severity','Replicate','BPFO_Hz', ...
    'RMS_m_s2','Std_m_s2','Peak_m_s2','Kurtosis','CrestFactor', ...
    'Amp1X_m_s2','Amp2X_m_s2','AmpBPFO_m_s2', ...
    'RPM_ActualMean','RPM_ActualStd','NominalLoad_Percent','MeanLoad_Percent', ...
    'Temperature_C','AmbientTemperature_C','SensorNoiseStd_m_s2', ...
    'SensorBandwidth_Hz','SensorMountingGain','TransientEvent','Run_ID', ...
    'OperatingPoint_ID','RMS_Y_m_s2','Peak_Y_m_s2','Kurtosis_Y', ...
    'Amp1X_Y_m_s2','Amp2X_Y_m_s2','AmpBPFO_Y_m_s2'});

save('rotor_fault_dataset.mat', 'signals', 'signalsY', 'metadata', ...
    'features', 't', 'cfg', '-v7');
writetable(features, 'rotor_fault_features.csv');

fprintf('\nCreated %d vibration records.\n', nRecords);
fprintf('Raw X/Y signals: rotor_fault_dataset.mat (signals, signalsY)\n');
fprintf('Feature table: rotor_fault_features.csv (%d columns)\n', width(features));
fprintf('Nominal RPM levels: %s\n', mat2str(cfg.rpmList));
fprintf('Nominal load levels: %s\n', mat2str(100*cfg.loadLevels));
fprintf('Transient load-step records: %d (%.1f%%)\n', ...
    sum(metaTransient), 100*mean(metaTransient));

%% 9. Plot representative signals and spectra
plotRPM = 1800;
plotLoad = 0.75;
plotRows = zeros(4,1);
plotNames = ["healthy", "unbalance", "misalignment", "bearing_outer"];
for k = 1:numel(plotNames)
    if plotNames(k) == "healthy"
        plotSeverity = 0;
    else
        plotSeverity = 3;
    end
    idx = find(metaRPM == plotRPM & ...
        abs(metaNominalLoad - plotLoad) < 1e-9 & ...
        metaCondition == plotNames(k) & metaSeverity == plotSeverity, 1, 'first');
    if isempty(idx)
        idx = find(metaRPM == plotRPM & metaCondition == plotNames(k) & ...
            metaSeverity == plotSeverity, 1, 'first');
    end
    plotRows(k) = idx;
end

figure('Name','Rotor fault simulation - X-axis signals','Color','w');
for k = 1:4
    subplot(4,1,k);
    plot(t, double(signals(plotRows(k),:)), 'LineWidth', 0.8);
    xlim([0 min(0.25, cfg.duration_s)]);
    ylabel('m/s^2');
    title(sprintf('%s, severity %d, nominal %d RPM', ...
        char(metaCondition(plotRows(k))), metaSeverity(plotRows(k)), plotRPM));
    if k == 4
        xlabel('Time [s]');
    end
end

figure('Name','Rotor fault simulation - X-axis spectra','Color','w');
for k = 1:4
    subplot(2,2,k);
    x = double(signals(plotRows(k),:));
    [freqHz, amp] = oneSidedAmplitude(x, cfg.fs);
    plot(freqHz, amp, 'LineWidth', 0.8);
    xlim([0 3000]);
    xlabel('Frequency [Hz]');
    ylabel('Peak amplitude [m/s^2]');
    title(sprintf('%s spectrum', char(metaCondition(plotRows(k)))));
end

%% Local functions
function [sensorX, sensorY, info] = simulateOneRecord( ...
    cfg, rpmCommand, condition, severity, nominalLoad, t)
    fs = cfg.fs;
    dt = 1/fs;

    % Generate a slowly varying speed profile and integrate it to shaft phase.
    rippleFrequency = cfg.speedRippleHzRange(1) + ...
        diff(cfg.speedRippleHzRange)*rand();
    speedPhase = 2*pi*rand();
    speedNoise = correlatedNoise(numel(t), cfg.coloredNoiseAR);
    speedFraction = cfg.speedRippleFraction * ...
        sin(2*pi*rippleFrequency*t + speedPhase) + ...
        cfg.speedRandomFraction*speedNoise;
    rpmProfile = rpmCommand*(1 + speedFraction);
    rpmProfile = max(0.5*rpmCommand, rpmProfile);
    shaftPhase = 2*pi*cumtrapz(t, rpmProfile/60);
    omegaProfile = 2*pi*rpmProfile/60;

    % Smooth load fluctuations about the requested operating load.
    loadNoise = correlatedNoise(numel(t), 0.985);
    loadRippleHz = 0.15 + 0.45*rand();
    loadProfile = nominalLoad + ...
        cfg.loadRippleFraction*sin(2*pi*loadRippleHz*t + 2*pi*rand()) + ...
        cfg.loadRandomFraction*loadNoise;

    % Some records contain a load-step event to represent changing demand.
    transientEvent = rand() < cfg.loadStepProbability;
    if transientEvent
        stepFraction = cfg.loadStepFractionRange(1) + ...
            diff(cfg.loadStepFractionRange)*rand();
        if rand() < 0.5
            stepFraction = -stepFraction;
        end
        stepTime = cfg.warmup_s + 0.20 + ...
            rand()*max(0.01, cfg.duration_s - 0.40);
        loadProfile(t >= stepTime) = loadProfile(t >= stepTime) + stepFraction;
    end
    loadProfile = min(1.0, max(0.10, loadProfile));
    keep = (t >= cfg.warmup_s);
    meanLoad = mean(loadProfile(keep));
    meanRPM = mean(rpmProfile(keep));
    stdRPM = std(rpmProfile(keep));

    % Ambient temperature and load-related temperature rise.
    ambientTemperature = cfg.ambientTemperatureRange_C(1) + ...
        diff(cfg.ambientTemperatureRange_C)*rand();
    temperature = ambientTemperature + ...
        cfg.temperatureRiseAtRatedLoad_C*meanLoad + 2.0*randn();
    temperature = min(90, max(10, temperature));

    % Rotor lateral stiffness and damping from the selected natural frequency.
    % Thermal coefficients and run-to-run scatters are explicit assumptions.
    mR = cfg.rotorMass_kg;
    thermalDelta = temperature - 50;
    kScale = 1 + cfg.stiffnessTempCoeff_perC*thermalDelta + ...
        cfg.machineStiffnessScatter*randn();
    cScale = 1 + cfg.dampingTempCoeff_perC*thermalDelta + ...
        cfg.machineDampingScatter*randn();
    kScale = max(0.80, kScale);
    cScale = max(0.60, cScale);
    wR = 2*pi*cfg.rotorNaturalFrequency_Hz;
    kR = mR*wR^2*kScale;
    cR = 2*cfg.rotorDampingRatio*sqrt(kR*mR)*cScale;

    % Local bearing/housing resonance parameters.
    mH = cfg.housingModalMass_kg;
    wH = 2*pi*cfg.housingResonance_Hz;
    kH = mH*wH^2*kScale;
    cH = 2*cfg.housingDampingRatio*sqrt(kH*mH)*cScale;

    % Three coordinates: rotor x, rotor y, and local housing mode.
    M = diag([mR, mR, mH]);
    C = diag([cR, cR, cH]);
    K = diag([kR, kR, kH]);

    % Residual unbalance is present in all classes. Faulted unbalance adds
    % eccentricity. The time-varying speed enters through instantaneous omega.
    eFault = 0;
    if condition == "unbalance"
        eFault = cfg.faultEccentricity_m(severity + 1);
    end
    assemblyGain = cfg.unbalanceAssemblyGainRange(1) + ...
        diff(cfg.unbalanceAssemblyGainRange)*rand();
    unbalanceProduct = cfg.unbalanceMass_kg * ...
        (cfg.residualEccentricity_m + eFault)*assemblyGain;
    unbalanceForce = unbalanceProduct .* omegaProfile.^2;
    Fx = unbalanceForce .* cos(shaftPhase);
    Fy = unbalanceForce .* sin(shaftPhase);
    Fh = zeros(size(t));

    % Simplified coupling-force representation of misalignment.
    if condition == "misalignment"
        A1 = cfg.misalignment1X_N(severity + 1);
        A2 = cfg.misalignment2X_N(severity + 1);
        alignmentGain = cfg.misalignmentGainRange(1) + ...
            diff(cfg.misalignmentGainRange)*rand();
        loadFactor = 0.65 + 0.70*loadProfile;
        Fx = Fx + alignmentGain*loadFactor.* ...
            (A1*cos(shaftPhase + 0.2) + A2*cos(2*shaftPhase + 0.6));
        Fy = Fy + alignmentGain*loadFactor.* ...
            (0.8*A1*sin(shaftPhase + 0.2) + ...
             0.6*A2*sin(2*shaftPhase + 0.6));
    end

    % Outer-race defect frequency from bearing geometry.
    nB = cfg.nRollingElements;
    d = cfg.ballDiameter_m;
    D = cfg.pitchDiameter_m;
    contactAngle = deg2rad(cfg.contactAngle_deg);
    bpfoOrder = (nB/2)*(1 - (d/D)*cos(contactAngle));
    bpfoInstant = bpfoOrder .* (rpmProfile/60);
    bpfoHz = mean(bpfoInstant(keep));

    % Repeated impacts occur at BPFO order and excite the local housing mode.
    if condition == "bearing_outer"
        impactPeak = cfg.bearingImpactPeak_N(severity + 1);
        Fh = makeImpactTrain(t, bpfoOrder, shaftPhase, loadProfile, ...
            impactPeak, cfg.impactSigma_s, cfg.bearingImpactGainRange);
    end

    % External machinery tones and dynamic load disturbance act on the rotor.
    ext1Amp = cfg.externalTone1ForceRange_N(1) + ...
        diff(cfg.externalTone1ForceRange_N)*rand();
    ext1Hz = cfg.externalTone1FrequencyRange_Hz(1) + ...
        diff(cfg.externalTone1FrequencyRange_Hz)*rand();
    ext2Amp = cfg.externalTone2ForceRange_N(1) + ...
        diff(cfg.externalTone2ForceRange_N)*rand();
    ext2Hz = cfg.externalTone2FrequencyRange_Hz(1) + ...
        diff(cfg.externalTone2FrequencyRange_Hz)*rand();
    externalPhase1 = 2*pi*rand();
    externalPhase2 = 2*pi*rand();
    Fx = Fx + ext1Amp*cos(2*pi*ext1Hz*t + externalPhase1) + ...
        ext2Amp*sin(2*pi*ext2Hz*t + externalPhase2);
    Fy = Fy + 0.8*ext1Amp*sin(2*pi*ext1Hz*t + externalPhase2) + ...
        0.7*ext2Amp*cos(2*pi*ext2Hz*t + externalPhase1);

    loadDisturbanceHz = cfg.externalLowFrequencyRange_Hz(1) + ...
        diff(cfg.externalLowFrequencyRange_Hz)*rand();
    loadDisturbancePhase = 2*pi*rand();
    dynamicLoad = cfg.loadDisturbanceForce_N .* loadProfile .* ...
        sin(2*pi*loadDisturbanceHz*t + loadDisturbancePhase);
    Fx = Fx + dynamicLoad;
    Fy = Fy + 0.7*dynamicLoad.*sin(2*pi*0.7*loadDisturbanceHz*t);

    force = [Fx; Fy; Fh];
    acceleration = newmarkAverageAcceleration(M, C, K, force, dt);

    % Two measured radial channels. The local housing mode is strongest on X.
    rawX = acceleration(1,:) + acceleration(3,:);
    rawY = acceleration(2,:) + 0.15*acceleration(3,:);

    % Sensor-to-sensor mounting gain and bandwidth variability.
    mountingGain = cfg.sensorMountingGainRange(1) + ...
        diff(cfg.sensorMountingGainRange)*rand();
    sensorBandwidth = cfg.sensorBandwidthRange_Hz(1) + ...
        diff(cfg.sensorBandwidthRange_Hz)*rand();
    rawX = mountingGain*rawX;
    rawY = mountingGain*rawY;
    rawX = firstOrderSensorResponse(rawX, fs, sensorBandwidth);
    rawY = firstOrderSensorResponse(rawY, fs, sensorBandwidth);

    % White + coloured sensor noise, slow baseline drift, quantisation, clipping.
    sensorNoiseStd = cfg.sensorNoiseStdRange_m_s2(1) + ...
        diff(cfg.sensorNoiseStdRange_m_s2)*rand();
    coloredX = correlatedNoise(numel(t), cfg.coloredNoiseAR);
    coloredY = correlatedNoise(numel(t), cfg.coloredNoiseAR);
    noiseX = sensorNoiseStd*(randn(size(t)) + ...
        cfg.coloredNoiseFraction*coloredX);
    noiseY = sensorNoiseStd*(randn(size(t)) + ...
        cfg.coloredNoiseFraction*coloredY);
    driftX = cfg.sensorDriftMax_m_s2*(2*rand()-1) * ...
        (t/max(t) - 0.5);
    driftY = cfg.sensorDriftMax_m_s2*(2*rand()-1) * ...
        (t/max(t) - 0.5);
    sensorX = rawX + noiseX + driftX;
    sensorY = rawY + noiseY + driftY;

    % Approximate finite sensor range and digital resolution.
    sensorX = min(cfg.sensorRange_m_s2, max(-cfg.sensorRange_m_s2, sensorX));
    sensorY = min(cfg.sensorRange_m_s2, max(-cfg.sensorRange_m_s2, sensorY));
    sensorX = round(sensorX/cfg.sensorResolution_m_s2)*cfg.sensorResolution_m_s2;
    sensorY = round(sensorY/cfg.sensorResolution_m_s2)*cfg.sensorResolution_m_s2;

    % Discard the startup transient; returned time series have cfg.duration_s.
    sensorX = sensorX(keep);
    sensorY = sensorY(keep);

    info.meanLoad = meanLoad;
    info.meanRPM = meanRPM;
    info.stdRPM = stdRPM;
    info.temperature_C = temperature;
    info.ambientTemperature_C = ambientTemperature;
    info.bpfoHz = bpfoHz;
    info.sensorNoiseStd = sensorNoiseStd;
    info.sensorBandwidth_Hz = sensorBandwidth;
    info.mountingGain = mountingGain;
    info.transientEvent = transientEvent;
end

function force = makeImpactTrain(t, bpfoOrder, shaftPhase, loadProfile, ...
    peakForce, sigma, gainRange)
    force = zeros(size(t));
    if bpfoOrder <= 0 || peakForce <= 0
        return;
    end

    % Bearing defect phase is proportional to shaft angle. Threshold crossings
    % produce impact times even while instantaneous RPM varies.
    defectCycles = bpfoOrder*(shaftPhase - shaftPhase(1))/(2*pi);
    impactIndices = find(diff(floor(defectCycles)) > 0) + 1;
    dt = t(2) - t(1);
    for j = 1:numel(impactIndices)
        impactIndex = impactIndices(j);
        ti = t(impactIndex);
        impactGain = gainRange(1) + diff(gainRange)*rand();
        loadFactor = 0.55 + 0.75*loadProfile(impactIndex);
        amplitude = peakForce*impactGain*loadFactor;
        i1 = max(1, floor((ti - 4*sigma - t(1))/dt) + 1);
        i2 = min(numel(t), ceil((ti + 4*sigma - t(1))/dt) + 1);
        idx = i1:i2;
        tau = t(idx) - ti;
        force(idx) = force(idx) + amplitude*exp(-0.5*(tau/sigma).^2);
    end
end

function acceleration = newmarkAverageAcceleration(M, C, K, force, dt)
    % Newmark-beta method: beta = 1/4, gamma = 1/2 (average acceleration).
    beta = 1/4;
    gamma = 1/2;
    nDof = size(M,1);
    nSamples = size(force,2);

    a0 = 1/(beta*dt^2);
    a1 = gamma/(beta*dt);
    a2 = 1/(beta*dt);
    a3 = 1/(2*beta) - 1;
    a4 = gamma/beta - 1;
    a5 = dt*(gamma/(2*beta) - 1);

    Keff = K + a0*M + a1*C;
    KeffInv = Keff \ eye(nDof);

    q = zeros(nDof,1);
    v = zeros(nDof,1);
    a = M \ (force(:,1) - C*v - K*q);

    acceleration = zeros(nDof,nSamples);
    acceleration(:,1) = a;
    for n = 1:nSamples-1
        rhs = force(:,n+1) + ...
            M*(a0*q + a2*v + a3*a) + ...
            C*(a1*q + a4*v + a5*a);
        qNew = KeffInv*rhs;
        aNew = a0*(qNew-q) - a2*v - a3*a;
        vNew = v + dt*((1-gamma)*a + gamma*aNew);

        q = qNew;
        v = vNew;
        a = aNew;
        acceleration(:,n+1) = a;
    end
end

function y = firstOrderSensorResponse(x, fs, cutoffHz)
    % Toolbox-free one-pole low-pass approximation of sensor/mount bandwidth.
    alpha = exp(-2*pi*cutoffHz/fs);
    y = zeros(size(x));
    y(1) = x(1);
    for k = 2:numel(x)
        y(k) = alpha*y(k-1) + (1-alpha)*x(k);
    end
end

function y = correlatedNoise(n, alpha)
    % Unit-standard-deviation AR(1) sequence; avoids toolbox dependencies.
    y = zeros(1,n);
    white = randn(1,n);
    scale = sqrt(max(eps, 1-alpha^2));
    for k = 2:n
        y(k) = alpha*y(k-1) + scale*white(k);
    end
    y = y - mean(y);
    y = y/(std(y) + eps);
end

function [freqHz, amplitude] = oneSidedAmplitude(x, fs)
    % Hann-windowed one-sided peak-amplitude spectrum with coherent-gain correction.
    x = x(:) - mean(x);
    n = numel(x);
    if n < 2
        freqHz = 0;
        amplitude = abs(x);
        return;
    end
    window = 0.5 - 0.5*cos(2*pi*(0:n-1)'/(n-1));
    X = fft(x.*window);
    nPositive = floor(n/2) + 1;
    amplitude = abs(X(1:nPositive))/sum(window);

    if rem(n,2) == 0
        if nPositive > 2
            amplitude(2:end-1) = 2*amplitude(2:end-1);
        end
    else
        amplitude(2:end) = 2*amplitude(2:end);
    end
    freqHz = (0:nPositive-1)'*(fs/n);
end

function value = amplitudeInBand(freqHz, amplitude, targetHz, duration_s, bandFraction)
    % Peak within an order band to tolerate small RPM-induced spectral smearing.
    resolutionHz = 1/duration_s;
    halfWidth = max(2*resolutionHz, abs(targetHz)*bandFraction);
    mask = abs(freqHz-targetHz) <= halfWidth;
    if any(mask)
        value = max(amplitude(mask));
    else
        [~, idx] = min(abs(freqHz-targetHz));
        value = amplitude(idx);
    end
end
