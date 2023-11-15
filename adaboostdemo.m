clear all
close all
clc

spambase = readmatrix("spambase.xlsx");
training = spambase(1:3450,:);
testing = spambase(3451:4601,:);

tmean = mean(training(:,1:57));
tstd = std(training(:,1:57));

training(:,1:57) = (training(:,1:57) - tmean)./ tstd;
%training = training(randperm(size(training,1)),:);

h = adaboost(training,100);

testing(:,1:57) = (testing(:,1:57) - tmean)./ tstd;

g = get_pred(h,testing);
nums = ones(size(g,1),1);
err_testing = get_error(g,testing(:,end),nums)

f = get_pred(h,training);
nums2 = ones(size(f,1),1);
err_training = get_error(f,training(:,end),nums2)


%[a,b]= randomize_and_sort(training);
%sorted = training(:,1:end-1);
%sorted(1:size(training,1)/10,:) = sort(training(1:size(training,1)/10,1:end-1));

function [sorted,newsample] = randomize_and_sort(sample)
    newsample = sample(randperm(size(sample,1)),:);
    sorted = newsample(:,1:end-1);
    for i = 1:10
        sorted(1+(size(newsample,1)/10)*(i-1):(size(newsample,1)/10)*i,:) = ...
        sort(newsample(1+(size(newsample,1)/10)*(i-1):(size(newsample,1)/10)*i,1:end-1));
    end
end

function a = get_alpha(error)
    a = 0.5 * log((1-error)/error);
end

function z = get_normalize(error)
    z = 2*sqrt(error*(1-error));
end

function e = get_error(y_true,y_pred,w)
    e = sum(w.*abs(y_true-y_pred))/sum(w);
end

function y_pred = get_pred(h,sample)
    y_pred = zeros(size(sample,1),1);
    for i = 1:size(h,1)
        y_pred = y_pred + h(i,3)*(sample(:,h(i,1))<h(i,2));
    end
end

function [row,column,low_error,stump,count, threshold] = get_stump(sorted,weights,sample)
    low_error = inf;
    for j = 1:size(sorted,2)
        classifier = zeros(size(sorted,1),1);
        curr_error = get_error(sample(:,end),classifier,weights);
        if(curr_error) < (low_error)
            low_error = curr_error;
            row = 1;
            column = j;
            stump = classifier;
            count = 0;
            threshold = sorted(1,j);
        end
        for i = 2:size(sorted,1)
            threshold = sorted(i,j);
            classifier = zeros(size(sorted,1),1);
            curr_count = 0;
            for k = 1:size(sorted,1)
                if sample(k,j) < threshold
                    classifier(k) = 1;
                    curr_count = curr_count+1;
                end
            end
            %classifier(i-1) = 1;
            curr_error = get_error(sample(:,end),classifier,weights);
            if(curr_error) < (low_error)
                low_error = curr_error;
                row=i;
                column = j;
                stump = classifier;
                count = curr_count;
            end
        end
        classifier = ones(size(sorted,1),1);
        curr_error = get_error(sample(:,end),classifier,weights);
        if(curr_error) < (low_error)
            low_error = curr_error;
            row = 1;
            column = j;
            stump = classifier;
            count = size(sorted,1);
            threshold = inf;
        end
    end
end

function res = adaboost(sample,T)
    [sorted,newsample] = randomize_and_sort(sample);
    %sorted = sort(sample(:,1:end-1));
    d = ones(size(sorted,1)/10,1);
    x=size(sorted,1);
    d = d/x;
    f = zeros(T,3);
    %Column 1 indicates column number
    %Column 2 indicates threshold t where
    % if x < t, h(x)=1, if x >= t, h(x) = 0
    %Column 3 indicates weight
    %f = zeros(size(sorted,1)/10,1);
    for i = 1:T
        index = mod(i-1,10);
        s_index = 1+(size(newsample,1)/10)*(index);
        e_index = (size(newsample,1)/10)*(index+1);
        curr_sorted = sorted(s_index:e_index,:);
        curr_sample = newsample(s_index:e_index,:);
        size(curr_sorted);
        size(curr_sample);
        [~,col,err,stump,~,thresh] = get_stump(curr_sorted,d,curr_sample);
        a = get_alpha(err);
        z = get_normalize(err);
        d = d.*exp(-a *abs(stump-curr_sample(:,end)))/z;
        f(i,1) = col;
        f(i,2) = thresh;
        f(i,3) = a;
    end
    res = f;
    %sign(f);
end
