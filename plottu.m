clear

color_matrix = [1,1,0.4;0,0.75,1;0,0.4,0.25;1,0.4,1;1,0.38,0.4];
subplot(1,3,1);
b=bar(Thick1);

set(gca,'XTickLabel',{'7.15','7.16','7.17','7.18','7.19'});
for i=1:5  
    text(i,Thick1(i),num2str(Thick1(i)),'VerticalAlignment','bottom','HorizontalAlignment','center');
    set(b(1),'facecolor',color_matrix(i,:))
end
ylabel('Thick1');

subplot(1,3,2);
b=bar(Thick2);
set(gca,'XTickLabel',{'7.15','7.16','7.17','7.18','7.19'});
for i=1:5  
    text(i,Thick2(i),num2str(Thick2(i)),'VerticalAlignment','bottom','HorizontalAlignment','center');
    set(b(1),'facecolor',color_matrix(i,:))
end
ylabel('Thick2');

subplot(1,3,3);
b=bar(Thick3);
set(gca,'XTickLabel',{'7.15','7.16','7.17','7.18','7.19'});
for i=1:5  
    text(i,Thick3(i),num2str(Thick3(i)),'VerticalAlignment','bottom','HorizontalAlignment','center');
    set(b(1),'facecolor',color_matrix(i,:))
end
ylabel('Thick3');