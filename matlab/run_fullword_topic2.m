function run_fullword_topic2
root = fileparts(fileparts(mfilename('fullpath')));
dataRoot = fullfile(root, 'data', 'speech_commands_v0.02');
resultsDir = fullfile(root, 'results');
figDir = fullfile(root, 'report', 'figures');

if ~exist(resultsDir, 'dir')
    mkdir(resultsDir);
end
if ~exist(figDir, 'dir')
    mkdir(figDir);
end

rng(0);
sampleRate = 16000;
params.frameLength = round(0.025 * sampleRate);
params.hopLength = round(0.01 * sampleRate);
params.energyFrameLength = round(0.02 * sampleRate);
params.energyHopLength = round(0.01 * sampleRate);
params.windowSamples = round(0.10 * sampleRate);
params.scanHop = round(0.02 * sampleRate);
params.nfftLocal = 512;
params.nfftGlobal = 2048;
params.cap.train = 150;
params.cap.val = 60;
params.cap.test = 120;
params.kGrid = [1 3 5 7 9];
params.maxKnnBank = 80;

targets = {'D', 'G', 'N', 'R'};
letters = {'d', 'g', 'n', 'r'};
[words, phonemes] = word_phonemes();

items = build_items(dataRoot, words, params);
items = add_features(items, sampleRate, params);

results = struct();
models = struct();
averageSpectra = zeros(numel(targets), params.nfftLocal / 2 + 1);

for targetIdx = 1:numel(targets)
    target = targets{targetIdx};
    letter = letters{targetIdx};
    fprintf('Evaluating target %s\n', letter);

    sets = build_balanced_sets(items, target, words, phonemes);
    banks = build_target_banks(sets.train.items, target, phonemes, params);
    averageSpectra(targetIdx, :) = banks.averageLocalSpectrum;

    globalResult = evaluate_global_fft(sets, banks.globalFftTemplate);
    stftMeanResult = evaluate_stft_mean_scan(sets, banks.spectrumMeanBank);
    stftKnnResult = evaluate_stft_knn_scan(sets, banks.spectrumKnnBank, params.kGrid);

    results.(letter).positiveWords = sets.train.positiveWords;
    results.(letter).globalFft = globalResult;
    results.(letter).stftMean = stftMeanResult;
    results.(letter).stftKnn = stftKnnResult;
    results.(letter).trainSize = numel(sets.train.items);
    results.(letter).valSize = numel(sets.val.items);
    results.(letter).testSize = numel(sets.test.items);

    models.(letter).globalFftTemplate = banks.globalFftTemplate;
    models.(letter).spectrumMeanBank = banks.spectrumMeanBank;
    models.(letter).spectrumKnnBank = banks.spectrumKnnBank;
    models.(letter).bestK = stftKnnResult.bestK;
    models.(letter).thresholds.globalFft = globalResult.threshold;
    models.(letter).thresholds.stftMean = stftMeanResult.threshold;
    models.(letter).thresholds.stftKnn = stftKnnResult.threshold;
end

summary = summarize_results(results, letters);
write_summary(fullfile(resultsDir, 'matlab_fullword_classical_metrics.txt'), summary, results, letters);
save(fullfile(resultsDir, 'matlab_fullword_classical_model.mat'), ...
    'models', 'params', 'targets', 'letters', 'words', 'phonemes', 'summary');
plot_average_spectra(fullfile(figDir, 'matlab_fullword_average_local_spectra.png'), ...
    averageSpectra, letters, sampleRate, params.nfftLocal);

disp(fileread(fullfile(resultsDir, 'matlab_fullword_classical_metrics.txt')));
end

function [words, phonemes] = word_phonemes()
words = {'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'five', 'forward', ...
    'four', 'go', 'house', 'marvin', 'nine', 'no', 'on', 'one', 'right', ...
    'seven', 'three', 'tree', 'up', 'wow', 'yes', 'zero'};
phonemes = struct();
phonemes.backward = {'B', 'AE', 'K', 'W', 'ER', 'D'};
phonemes.bed = {'B', 'EH', 'D'};
phonemes.bird = {'B', 'ER', 'D'};
phonemes.cat = {'K', 'AE', 'T'};
phonemes.dog = {'D', 'AO', 'G'};
phonemes.down = {'D', 'AW', 'N'};
phonemes.five = {'F', 'AY', 'V'};
phonemes.forward = {'F', 'AO', 'R', 'W', 'ER', 'D'};
phonemes.four = {'F', 'AO', 'R'};
phonemes.go = {'G', 'OW'};
phonemes.house = {'HH', 'AW', 'S'};
phonemes.marvin = {'M', 'AA', 'R', 'V', 'IH', 'N'};
phonemes.nine = {'N', 'AY', 'N'};
phonemes.no = {'N', 'OW'};
phonemes.on = {'AA', 'N'};
phonemes.one = {'W', 'AH', 'N'};
phonemes.right = {'R', 'AY', 'T'};
phonemes.seven = {'S', 'EH', 'V', 'AH', 'N'};
phonemes.three = {'TH', 'R', 'IY'};
phonemes.tree = {'T', 'R', 'IY'};
phonemes.up = {'AH', 'P'};
phonemes.wow = {'W', 'AW'};
phonemes.yes = {'Y', 'EH', 'S'};
phonemes.zero = {'Z', 'IH', 'R', 'OW'};
end

function items = build_items(dataRoot, words, params)
template = empty_item();
items.train = template([]);
items.val = template([]);
items.test = template([]);

for wordIdx = 1:numel(words)
    word = words{wordIdx};
    files = dir(fullfile(dataRoot, word, '*.wav'));
    [~, order] = sort({files.name});
    files = files(order);
    counts.train = 0;
    counts.val = 0;
    counts.test = 0;
    for fileIdx = 1:numel(files)
        speakerId = extractBefore(files(fileIdx).name, '_nohash_');
        split = speaker_split(char(speakerId));
        if counts.(split) >= params.cap.(split)
            continue;
        end
        counts.(split) = counts.(split) + 1;
        newItem = empty_item();
        newItem.word = word;
        newItem.path = fullfile(files(fileIdx).folder, files(fileIdx).name);
        items.(split)(end + 1) = newItem;
    end
end
end

function item = empty_item()
item = struct('word', '', 'path', '', 'activeBounds', [], 'windowStarts', [], ...
    'scanSpectra', [], 'globalFft', []);
end

function split = speaker_split(speakerId)
bucket = mod(hex2dec(speakerId), 100);
if bucket < 20
    split = 'test';
elseif bucket < 30
    split = 'val';
else
    split = 'train';
end
end

function items = add_features(items, sampleRate, params)
splitNames = {'train', 'val', 'test'};
total = numel(items.train) + numel(items.val) + numel(items.test);
count = 0;
for splitIdx = 1:numel(splitNames)
    split = splitNames{splitIdx};
    for itemIdx = 1:numel(items.(split))
        count = count + 1;
        [signal, fs] = audioread(items.(split)(itemIdx).path);
        if fs ~= sampleRate
            error('Unexpected sample rate in %s', items.(split)(itemIdx).path);
        end
        signal = normalize_signal(signal(:, 1));
        activeBounds = active_bounds(signal, sampleRate, params);
        [windowStarts, scanSpectra] = extract_scan_spectra(signal, activeBounds, params);
        items.(split)(itemIdx).activeBounds = activeBounds;
        items.(split)(itemIdx).windowStarts = windowStarts;
        items.(split)(itemIdx).scanSpectra = scanSpectra;
        items.(split)(itemIdx).globalFft = global_fft_feature(signal, activeBounds, params);
        if mod(count, 500) == 0 || count == total
            fprintf('Cached %d/%d utterances\n', count, total);
        end
    end
end
end

function signal = normalize_signal(signal)
signal = signal(:)';
signal = signal - mean(signal);
peak = max(abs(signal));
if peak > 0
    signal = signal / peak;
end
end

function bounds = active_bounds(signal, sampleRate, params)
frames = frame_signal(signal, params.energyFrameLength, params.energyHopLength);
energy = mean(frames .^ 2, 2);
threshold = max(1e-6, 0.10 * max(energy));
active = find(energy >= threshold);
if isempty(active)
    bounds = [1 numel(signal)];
    return;
end
startSample = max(1, (active(1) - 1) * params.energyHopLength + 1 - round(0.01 * sampleRate));
endSample = min(numel(signal), (active(end) - 1) * params.energyHopLength + ...
    params.energyFrameLength + round(0.01 * sampleRate));
if endSample - startSample + 1 < params.windowSamples
    endSample = min(numel(signal), startSample + params.windowSamples - 1);
end
bounds = [startSample endSample];
end

function frames = frame_signal(signal, frameLength, hopLength)
if numel(signal) < frameLength
    signal = [signal zeros(1, frameLength - numel(signal))];
end
nFrames = 1 + floor((numel(signal) - frameLength) / hopLength);
frames = zeros(nFrames, frameLength);
for frameIdx = 1:nFrames
    startIdx = (frameIdx - 1) * hopLength + 1;
    frames(frameIdx, :) = signal(startIdx:startIdx + frameLength - 1);
end
end

function [windowStarts, scanSpectra] = extract_scan_spectra(signal, activeBounds, params)
startSample = activeBounds(1);
endSample = activeBounds(2);
if endSample - startSample + 1 <= params.windowSamples
    windowStarts = startSample;
else
    lastStart = endSample - params.windowSamples + 1;
    windowStarts = startSample:params.scanHop:lastStart;
    if windowStarts(end) ~= lastStart
        windowStarts(end + 1) = lastStart;
    end
end
scanSpectra = zeros(numel(windowStarts), params.nfftLocal / 2 + 1);
for idx = 1:numel(windowStarts)
    segment = signal(windowStarts(idx):min(numel(signal), windowStarts(idx) + params.windowSamples - 1));
    if numel(segment) < params.windowSamples
        segment = [segment zeros(1, params.windowSamples - numel(segment))];
    end
    magnitude = stft_magnitude(segment, params);
    scanSpectra(idx, :) = average_spectrum_feature(magnitude);
end
end

function magnitude = stft_magnitude(segment, params)
frames = frame_signal(segment, params.frameLength, params.hopLength);
frames = frames .* repmat(local_hamming(params.frameLength), size(frames, 1), 1);
spectra = abs(fft(frames, params.nfftLocal, 2));
magnitude = spectra(:, 1:(params.nfftLocal / 2 + 1));
end

function feature = average_spectrum_feature(magnitude)
feature = mean(log1p(magnitude), 1);
feature = feature / (norm(feature) + 1e-8);
end

function feature = global_fft_feature(signal, activeBounds, params)
active = signal(activeBounds(1):activeBounds(2));
windowed = active .* local_hamming(numel(active));
magnitude = abs(fft(windowed, params.nfftGlobal));
magnitude = magnitude(1:(params.nfftGlobal / 2 + 1));
feature = log1p(magnitude);
feature = feature / (norm(feature) + 1e-8);
end

function sets = build_balanced_sets(items, target, words, phonemes)
positiveWords = {};
for wordIdx = 1:numel(words)
    phones = phonemes.(words{wordIdx});
    if any(strcmp(phones, target))
        positiveWords{end + 1} = words{wordIdx}; %#ok<AGROW>
    end
end

splitNames = {'train', 'val', 'test'};
for splitIdx = 1:numel(splitNames)
    split = splitNames{splitIdx};
    positiveMask = false(1, numel(items.(split)));
    for itemIdx = 1:numel(items.(split))
        positiveMask(itemIdx) = any(strcmp(positiveWords, items.(split)(itemIdx).word));
    end
    positiveIdx = find(positiveMask);
    negativeIdx = find(~positiveMask);
    negativeIdx = negativeIdx(randperm(numel(negativeIdx)));
    negativeIdx = negativeIdx(1:numel(positiveIdx));
    selectedIdx = [positiveIdx negativeIdx];
    selectedIdx = selectedIdx(randperm(numel(selectedIdx)));
    selectedItems = items.(split)(selectedIdx);
    labels = zeros(numel(selectedItems), 1);
    for itemIdx = 1:numel(selectedItems)
        labels(itemIdx) = any(strcmp(positiveWords, selectedItems(itemIdx).word));
    end
    sets.(split).items = selectedItems;
    sets.(split).labels = labels;
    sets.(split).positiveWords = positiveWords;
end
end

function banks = build_target_banks(trainItems, target, phonemes, params)
spectraByPosition.initial = [];
spectraByPosition.medial = [];
spectraByPosition.final = [];
allSpectra = [];
globalFeatures = [];

for itemIdx = 1:numel(trainItems)
    phones = phonemes.(trainItems(itemIdx).word);
    occurrenceIdx = find(strcmp(phones, target));
    if isempty(occurrenceIdx)
        continue;
    end
    globalFeatures(end + 1, :) = trainItems(itemIdx).globalFft; %#ok<AGROW>
    for idx = 1:numel(occurrenceIdx)
        occurrence = occurrenceIdx(idx);
        windowIdx = pick_occurrence_window(trainItems(itemIdx), occurrence, numel(phones), params);
        spectrum = trainItems(itemIdx).scanSpectra(windowIdx, :);
        position = occurrence_position(occurrence, numel(phones));
        spectraByPosition.(position)(end + 1, :) = spectrum; %#ok<AGROW>
        allSpectra(end + 1, :) = spectrum; %#ok<AGROW>
    end
end

spectrumMeanBank = [];
positions = {'initial', 'medial', 'final'};
for posIdx = 1:numel(positions)
    position = positions{posIdx};
    if ~isempty(spectraByPosition.(position))
        template = mean(spectraByPosition.(position), 1);
        template = template / (norm(template) + 1e-8);
        spectrumMeanBank(end + 1, :) = template; %#ok<AGROW>
    end
end

banks.spectrumMeanBank = spectrumMeanBank;
banks.spectrumKnnBank = select_representative_bank(allSpectra, params.maxKnnBank);
globalTemplate = mean(globalFeatures, 1);
banks.globalFftTemplate = globalTemplate / (norm(globalTemplate) + 1e-8);
banks.averageLocalSpectrum = mean(allSpectra, 1);
end

function windowIdx = pick_occurrence_window(item, occurrenceIdx, nPhones, params)
activeLength = max(item.activeBounds(2) - item.activeBounds(1) + 1, params.windowSamples);
targetCenter = item.activeBounds(1) + round(((occurrenceIdx - 0.5) / nPhones) * activeLength);
centers = item.windowStarts + floor(params.windowSamples / 2);
[~, windowIdx] = min(abs(centers - targetCenter));
end

function position = occurrence_position(occurrenceIdx, nPhones)
ratio = (occurrenceIdx - 0.5) / nPhones;
if ratio <= 1 / 3
    position = 'initial';
elseif ratio >= 2 / 3
    position = 'final';
else
    position = 'medial';
end
end

function bank = select_representative_bank(features, maxSize)
if size(features, 1) <= maxSize
    bank = features;
    return;
end
centroid = mean(features, 1);
centroid = centroid / (norm(centroid) + 1e-8);
similarity = features * centroid';
[~, order] = sort(similarity, 'descend');
bank = features(order(1:maxSize), :);
end

function result = evaluate_global_fft(sets, template)
valScores = zeros(numel(sets.val.items), 1);
for itemIdx = 1:numel(sets.val.items)
    valScores(itemIdx) = sets.val.items(itemIdx).globalFft * template';
end
[threshold, ~] = choose_threshold(valScores, sets.val.labels);
testScores = zeros(numel(sets.test.items), 1);
for itemIdx = 1:numel(sets.test.items)
    testScores(itemIdx) = sets.test.items(itemIdx).globalFft * template';
end
result = binary_metrics(testScores, sets.test.labels, threshold);
result.threshold = threshold;
end

function result = evaluate_stft_mean_scan(sets, bank)
valScores = scan_scores(sets.val.items, bank);
[threshold, ~] = choose_threshold(valScores, sets.val.labels);
testScores = scan_scores(sets.test.items, bank);
result = binary_metrics(testScores, sets.test.labels, threshold);
result.threshold = threshold;
end

function result = evaluate_stft_knn_scan(sets, bank, kGrid)
bestF1 = -Inf;
bestK = kGrid(1);
bestThreshold = 0;
for k = kGrid
    valScores = knn_scan_scores(sets.val.items, bank, k);
    [threshold, f1] = choose_threshold(valScores, sets.val.labels);
    if f1 > bestF1
        bestF1 = f1;
        bestK = k;
        bestThreshold = threshold;
    end
end
testScores = knn_scan_scores(sets.test.items, bank, bestK);
result = binary_metrics(testScores, sets.test.labels, bestThreshold);
result.threshold = bestThreshold;
result.bestK = bestK;
end

function scores = scan_scores(items, bank)
scores = zeros(numel(items), 1);
for itemIdx = 1:numel(items)
    similarities = items(itemIdx).scanSpectra * bank';
    scores(itemIdx) = max(similarities(:));
end
end

function scores = knn_scan_scores(items, bank, k)
scores = zeros(numel(items), 1);
k = min(k, size(bank, 1));
for itemIdx = 1:numel(items)
    similarities = items(itemIdx).scanSpectra * bank';
    sorted = sort(similarities, 2, 'descend');
    topk = mean(sorted(:, 1:k), 2);
    scores(itemIdx) = max(topk);
end
end

function [threshold, bestF1] = choose_threshold(scores, labels)
candidates = unique(scores);
candidates = [candidates(1) - 1e-6; candidates; candidates(end) + 1e-6];
bestF1 = -Inf;
threshold = candidates(1);
for idx = 1:numel(candidates)
    metrics = binary_metrics(scores, labels, candidates(idx));
    if metrics.f1 > bestF1
        bestF1 = metrics.f1;
        threshold = candidates(idx);
    end
end
end

function metrics = binary_metrics(scores, labels, threshold)
pred = scores >= threshold;
tp = sum(pred == 1 & labels == 1);
fp = sum(pred == 1 & labels == 0);
fn = sum(pred == 0 & labels == 1);
tn = sum(pred == 0 & labels == 0);
precision = tp / (tp + fp + 1e-8);
recall = tp / (tp + fn + 1e-8);
f1 = 2 * precision * recall / (precision + recall + 1e-8);
accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8);
metrics.precision = precision;
metrics.recall = recall;
metrics.f1 = f1;
metrics.accuracy = accuracy;
metrics.tp = tp;
metrics.fp = fp;
metrics.fn = fn;
metrics.tn = tn;
end

function summary = summarize_results(results, letters)
methods = {'globalFft', 'stftMean', 'stftKnn'};
for methodIdx = 1:numel(methods)
    method = methods{methodIdx};
    accuracy = zeros(numel(letters), 1);
    precision = zeros(numel(letters), 1);
    recall = zeros(numel(letters), 1);
    f1 = zeros(numel(letters), 1);
    for letterIdx = 1:numel(letters)
        letter = letters{letterIdx};
        metric = results.(letter).(method);
        accuracy(letterIdx) = metric.accuracy;
        precision(letterIdx) = metric.precision;
        recall(letterIdx) = metric.recall;
        f1(letterIdx) = metric.f1;
    end
    summary.(method).macroAccuracy = mean(accuracy);
    summary.(method).macroPrecision = mean(precision);
    summary.(method).macroRecall = mean(recall);
    summary.(method).macroF1 = mean(f1);
end
end

function write_summary(path, summary, results, letters)
fid = fopen(path, 'w');
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, 'MATLAB full-word classical voiced-consonant detection summary\n\n');
fprintf(fid, 'Global FFT template: macro F1=%.4f, macro accuracy=%.4f\n', ...
    summary.globalFft.macroF1, summary.globalFft.macroAccuracy);
fprintf(fid, 'Local STFT mean-template scan: macro F1=%.4f, macro accuracy=%.4f\n', ...
    summary.stftMean.macroF1, summary.stftMean.macroAccuracy);
fprintf(fid, 'Local STFT k-NN scan: macro F1=%.4f, macro accuracy=%.4f\n\n', ...
    summary.stftKnn.macroF1, summary.stftKnn.macroAccuracy);
fprintf(fid, 'Per-letter F1\n');
for letterIdx = 1:numel(letters)
    letter = letters{letterIdx};
    fprintf(fid, '%s: global_fft=%.4f, stft_mean=%.4f, stft_knn=%.4f\n', letter, ...
        results.(letter).globalFft.f1, results.(letter).stftMean.f1, ...
        results.(letter).stftKnn.f1);
end
end

function plot_average_spectra(path, averageSpectra, letters, sampleRate, nfft)
freqAxis = linspace(0, sampleRate / 2, nfft / 2 + 1);
figure('Visible', 'off');
hold on;
for idx = 1:numel(letters)
    curve = 20 * log10(averageSpectra(idx, :) / max(averageSpectra(idx, :)) + 1e-8);
    plot(freqAxis, curve, 'LineWidth', 1.6, 'DisplayName', letters{idx});
end
xlim([0 4000]);
ylim([-40 1]);
grid on;
xlabel('Frequency (Hz)');
ylabel('Normalized magnitude (dB)');
title('MATLAB Average Local Spectra of Target Consonants');
legend('Location', 'best');
saveas(gcf, path);
close(gcf);
end

function w = local_hamming(n)
if n <= 1
    w = ones(1, n);
    return;
end
k = 0:(n - 1);
w = 0.54 - 0.46 * cos(2 * pi * k / (n - 1));
end
