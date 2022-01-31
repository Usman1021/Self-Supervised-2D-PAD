function [H,s,ang,t,R] = cvexTformToSRT(H)
R = H(1:2,1:2);
t = H(3, 1:2);
ang = mean([atan2(R(2),R(1)) atan2(-R(3),R(4))]);
s = mean(R([1 4])/cos(ang));
% Reconstitute new s-R-t transform:
R = [cos(ang) -sin(ang); sin(ang) cos(ang)];
H = [[s*R; t], [0 0 1]'];
end