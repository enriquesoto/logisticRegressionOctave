data = load('car.txt');
X = data(:, [1, 2,3,4,5,6]); y = data(:, 7);
tamY=size(y);
temp = zeros(tamY,4);
for i =1:tamY
	for j=1:size(temp,2)	
		if y(i)==j
		      temp(i,j)=1;
		else
		      temp(i,j)=0;
		endif
	endfor
endfor