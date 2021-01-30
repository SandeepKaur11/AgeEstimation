function [cumulative_error] = cal_cum_err(yhat_test, ytest)
    sorted_aber = sort(abs(yhat_test-ytest));
    cumulative_error = zeros(ceil(sorted_aber(end)), 1);
    c_i = 1;
    for ae_idx = 1:size(sorted_aber, 1)
        while sorted_aber(ae_idx) > c_i
            c_i = c_i + 1;
            cumulative_error(c_i) = cumulative_error(c_i) + cumulative_error(c_i-1);
        end
        cumulative_error(c_i) = cumulative_error(c_i) + 1;
    end
    cumulative_error = cumulative_error/size(ytest, 1);
end