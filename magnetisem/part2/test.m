folder = "/Users/user/Documents/semster_c/courses/lab/magnetisem/part2/week3/ex2/backward/"
bwValuesLow = [];
bwValuesHigh = [];
bwValues = [];
grayThreshes = [];
voltages = [];

bmpFiles = dir(folder + "*.bmp");

% callibrate
for i = 1 : size(bmpFiles,1)
    image = imread(strcat(bmpFiles(i).folder ,'/' ,bmpFiles(i).name));
    grayThreshes(end+1) = graythresh(image);
end

% calculating the systersis
g = grayThreshes;
grayThreshes(find(grayThreshes > max(grayThreshes)-0.02)) = [];
grayThreshes(find(grayThreshes < min(grayThreshes)+0.02)) = [];
threshold = mean(grayThreshes);
thresholdSTD = std(grayThreshes);

for i = 1 : size(bmpFiles,1)
    image = imread(strcat(bmpFiles(i).folder ,'/' ,bmpFiles(i).name));
    [filepath,name,ext] = fileparts(bmpFiles(i).name);
    voltages(end+1) = str2num(name);
    imageBW = imbinarize(image,threshold);
    bwValues(end+1) = mean(imageBW,'all');
    imageBWLow = imbinarize(image,threshold + thresholdSTD);
    bwValuesLow(end+1) = mean(imageBWLow,'all');
    imageBWHigh = imbinarize(image,threshold - thresholdSTD);
    bwValuesHigh(end+1) = mean(imageBWHigh,'all');
end

bwDeltaPos = bwValuesHigh - bwValues;
bwDeltaNeg = bwValues - bwValuesLow;
