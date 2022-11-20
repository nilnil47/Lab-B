% Enter folder name, don't forget the \ at the end of the path
folder = "/Users/user/Documents/semster_c/courses/lab/magnetisem/part2/week3/ex2/forward/";
bmpFiles = ls(folder + "*.bmp");
% % Batch 1
% firstPicNum = 4;
% maxVoltage = 4;
% Batch 2
firstPicNum = 90;
maxVoltage = 5;
% % Batch 3
% firstPicNum = 240;
% maxVoltage = 5;

bwValuesLow = [];
bwValuesHigh = [];
bwValues = [];
grayThreshes = [];
voltages = -maxVoltage:0.2:maxVoltage;

for i = firstPicNum : size(bmpFiles,1) + firstPicNum - 1
    image = imread(strcat(folder,"Capture_",int2str(i),".bmp"));
    grayThreshes(end+1) = graythresh(image);
%     if mod(i,7) == 0
%         figure
%        imshowpair(image,imageBW,'montage') 
%     end
end
g = grayThreshes;
grayThreshes(find(grayThreshes > max(grayThreshes)-0.02)) = [];
grayThreshes(find(grayThreshes < min(grayThreshes)+0.02)) = [];
threshold = mean(grayThreshes);
thresholdSTD = std(grayThreshes);

for i = firstPicNum : size(bmpFiles,1) + firstPicNum - 1
    image = imread(strcat(folder,"Capture_",int2str(i),".bmp"));
    imageBW = imbinarize(image,threshold);
    bwValues(end+1) = mean(imageBW,'all');
    imageBWLow = imbinarize(image,threshold + thresholdSTD);
    bwValuesLow(end+1) = mean(imageBWLow,'all');
    imageBWHigh = imbinarize(image,threshold - thresholdSTD);
    bwValuesHigh(end+1) = mean(imageBWHigh,'all');
end
bwDeltaPos = bwValuesHigh - bwValues;
bwDeltaNeg = bwValues - bwValuesLow;
bwFirst = bwValues(1:size(voltages,2));
bwSecond = bwValues(size(voltages,2)+1:end);
bwDeltaPosFirst = bwDeltaPos(1:size(voltages,2));
bwDeltaPosSecond = bwDeltaPos(size(voltages,2)+1:end);
bwDeltaNegFirst = bwDeltaNeg(1:size(voltages,2));
bwDeltaNegSecond = bwDeltaNeg(size(voltages,2)+1:end);

e1 = errorbar(-voltages(1:size(bwFirst,2)), bwFirst - 0.5, bwDeltaNegFirst, bwDeltaPosFirst, '*');
hold on
e2 = errorbar(voltages(2:size(bwSecond,2)+1), bwSecond - 0.5, bwDeltaNegSecond, bwDeltaPosSecond, '*');

% p1 = plot(-voltages(1:size(bwFirst,2)), bwFirst - 0.5, '*');
% hold on
% p2 = plot(voltages(2:size(bwSecond,2)+1), bwSecond - 0.5, '*');
e1.LineWidth = 0.9;
e2.LineWidth = 0.9;
e1.MarkerSize = 10;
e2.MarkerSize = 10;

xlabel('Voltage [V]','FontSize', 55);
ylabel('Normalized Average Brightness','FontSize', 55);
set(gca,'FontSize',25);
grid on;

figure
p3 = plot(g,'*');
xlabel('# Of Picture','FontSize', 25);
ylabel('Gray Threshold','FontSize', 25);
set(gca,'FontSize',25);

