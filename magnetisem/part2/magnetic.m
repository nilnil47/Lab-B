forwardFolder = "/Users/user/Documents/semster_c/courses/lab/magnetisem/part2/week3/ex2/forward/";
backwardFolder = "/Users/user/Documents/semster_c/courses/lab/magnetisem/part2/week3/ex2/backward/";

[voltagesFirst, bwFirst, bwDeltaNegFirst, bwDeltaPosFirst, g1] = getBw(forwardFolder);
[voltagesSecond, bwSecond , bwDeltaNegSecond, bwDeltaPosSecond, g2] = getBw(backwardFolder);

e1 = errorbar(voltagesFirst, bwFirst, bwDeltaNegFirst, bwDeltaPosFirst, '*');
hold on
e2 = errorbar(-voltagesSecond, bwSecond, bwDeltaNegSecond, bwDeltaPosSecond, '*');

e1.LineWidth = 0.9;
e2.LineWidth = 0.9;
e1.MarkerSize = 10;
e2.MarkerSize = 10;

xlabel('Voltage [V]','FontSize', 55);
ylabel('Normalized Average Brightness','FontSize', 55);
set(gca,'FontSize',25);
grid on;

figure
p3 = plot([g1, g2],'*');
xlabel('# Of Picture','FontSize', 25);
ylabel('Gray Threshold','FontSize', 25);
set(gca,'FontSize',25);

