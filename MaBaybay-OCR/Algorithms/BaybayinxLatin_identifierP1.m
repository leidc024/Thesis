% ---------------------------------------------------------------------------------------%
% OCR Type II - Distinguishing Baybayin words from Latin                                 %                           
% by Rodney Pino, Renier Mendoza, and Rachelle Sambayan                                  %
% Programmed by Rodney Pino at University of the Philippines - Diliman                   %
% Programming dates: September 2023 to January 2024                                      % 
% ---------------------------------------------------------------------------------------%

function [output1,WBB,A,output,Baybayin_counter,message1]=BaybayinxLatin_identifierP1(LvsB_classifier_00125,a)
% iniTime = clock;
% limit   = 180;  %transliteration limit time (in seconds)
tic; %ticBytes(gcp);
format compact
clf(); %close all;

%Start of preprocessing
A=imread(a);
%[~,v2]=c2bw_orig(A);        % rgb to binary image conversion
%size(v2);               % image size information
v2=rgb2gray(A);   % removing small or noise components

%=======Acquisition of Text Properties using the built-in ocr function=======%

% ================================================================= %
% Note:                                                             %
% The template is set this way for the reason that the ocr function %
% may still produce the text properties even if the languages used  %
% cannot be applied to the input characters. This is due to the     %
% novelty of Baybayin OCR and there are no available Baybayin       %
% OCR at the time of making this project                            %
% ------------------------------------------------------------------%

text1=ocr(A, 'Language','Tagalog','Language','Malay','TextLayout','Block');

RW  = text1.Words;
if isempty(RW)
text1=ocr(A, 'Language','Tagalog','Language','Malay','TextLayout','none');
RW  = text1.Words;
if isempty(RW)
    error('Was not able to read the text. Sorry');
end
end 

%----------------------------------------------------------------------------%

%======================== Word Segmentation Process ========================%

output=zeros(1,length(RW)); %preallocating the number of words found
WBB = text1.WordBoundingBoxes;
[row, ~]=size(WBB);


Latin=0; Baybayin=0;

Modused1=LvsB_classifier_00125; Workers=12;

parfor (i=1:row, Workers)
% if i==2
%    j=11;
% end    
%     
    P=WBB(i,:);
    PP=[P(1)-1 P(2)-1 P(3)+1 P(4)+1];
    PPP=imcrop(v2, PP);
    
%edits-------------------------------
       PPP=im2single(PPP);
       PPP=imsegkmeans(PPP,2); A2=mode(mode(PPP));
       PPP=(PPP~=A2);   
%---------------------------------------------     
    PPP=bwareaopen(PPP,13);     
    
    S=regionprops(PPP,'basic');
s=struct2cell(S);
[~,cs]=size(s);
CC=zeros(1, cs);
BBs=s(3,:);
for j=1:cs
    C=s{2,j};
    CC(j)=C(1);
end
C3=sort(CC);
CCC=sort(CC);

[~, colPPP]=size(PPP);
threshold_val=colPPP/(length(CCC)+1);

%----------------- computing each character bounding boxes -----------------%
for jj=1:cs-1
    for k=jj+1:cs
    if abs(CCC(jj)-CCC(k))<threshold_val
        CCC(k)=0;
    end
    end
end

K=part_of_the_character_finder(CCC);   %determines the component(s) of a Baybayin character
                                       %from (each) word bounding boxes obtained from 
                                       %OCR function

CBB=cell(1,length(K));
kcol1=0;
for q=1:length(K)
    
    [~, kcol]=size(K{q});
    C5=C3(1,1+kcol1:kcol1+kcol);
    
    kcol1=kcol1+kcol;
    
    Q=zeros(1,length(C5));
    BB=[];
    for qq=1:length(C5)
        [~, idx]=find(CC==C5(qq));
        if length(idx)>1
            idx=idx(1);
        end
        Q(qq)=idx;
        BB=cat(1,BB,BBs{Q(qq)});
    end

    BB1=BB(:,1);
    BB2=BB(:,2);
    BB3=BB(:,3);
    BB4=BB(:,4);
    
    AA1=min(BB2); AA2=max(BB2);
    AA3=abs(AA1-AA2);
    [~, idx]=max(BB2);
    AA4=AA3+BB4(idx);
   
    CBB{q}=[min(BB1) min(BB2) max(BB3) AA4];
    
    
end
%---------------------------------------------------------------------------%

%======================== Baybayin and Latin Script Recognition ========================%

[~, m]=size(CBB);
L_B=zeros(1,m);

% if i==2
%    j=11;
% end

for ii=1:m                                              %word extraction
    crop=CBB{1,ii};
    crop=[crop(1)-1 crop(2)-1 crop(3)+1 crop(4)+1];
    character=imcrop(PPP,crop);                         %Isolating the character
    character=bwareaopen(character,70);
    character=imdilate(character, strel('disk',1));
%    if i==3
%        j=1;
%    end
%   imshow(character);
    LvsB_label=LVSB(Modused1,character);                         %Latin and Baybayin character function classifier
    L_B(1,ii)=LvsB_label;                               %Each categorization result is stored
end  

      L_B=sign(sum(L_B));                               %The word script depends on the sign of the sum of the categorization result
    if L_B==1 || L_B==0                                 %Latin if 1 or 0
        Latin=Latin+1;
        output(i)=1;
    else
        Baybayin=Baybayin+1;                            %If -1, it is Baybayin
        output(i)=-1;
    end
    
end

%------------------------------ Setting the output image/s ------------------------------%


% Figg=figure('visible','off');  imshow(a); pause(0.1)

hold on

% pause(0.1)

Baybayin_counter=[]; ff=cell(1,row); paramss=[];
    for j=1:row
        
        if output(j)==1
            ff{j}=@() rectangle('Position', WBB(j,:));          
            paramss=cat(2,paramss,{{'edgecolor','b'}});         %Latin words are bounded by blue boxes
            
        else
            Baybayin_counter=cat(1,Baybayin_counter,j);
            ff{j}=@() rectangle('Position',WBB(j,:));
            paramss=cat(2,paramss,{{'edgecolor','r'}});         %Baybayin words are bounded by red boxes
        end
    end
%pause(0.0001);

%Figg1=getframe(Figg); output1=Figg1.cdata;
output1=insertInImage(A,ff,paramss);



%message=
if sum(output)==length(output)
message1=sprintf('Total words detected: %d.\n Baybayin: %d (bounded by red boxes)\n Latin: %d (bounded by blue boxes)\n \n No Baybayin words detected.',Latin+Baybayin, Baybayin, Latin); %output caption    
else
message1=sprintf('Total words detected: %d.\n Baybayin: %d (bounded by red boxes)\n Latin: %d (bounded by blue boxes)\n \n Proceeding to transliterations, please wait.',Latin+Baybayin, Baybayin, Latin); %output caption
end

 
toc;
end