% ---------------------------------------------------------------------------------------%
% OCR Type I - Boxing all Baybayin words detected                                        %                           
% by Rodney Pino, Renier Mendoza, and Rachelle Sambayan                                  %
% Programmed by Rodney Pino at University of the Philippines - Diliman                   %
% Programming dates: September 2023 to January 2024                                      % 
% ---------------------------------------------------------------------------------------%

function [output1,WBB,A,output,Baybayin_counter,message1]=Baybayin_identifier_onlyP1(a)
% iniTime = clock;
% limit   = 180;  %transliteration limit time (in seconds)
tic; %ticBytes(gcp);
format compact
clf(); %close all;

%Start of preprocessing
A=imread(a);
% [~,v2]=c2bw_orig(A);        % rgb to binary image conversion
% size(v2);               % image size information
% v2=bwareaopen(v2,10);   % removing small or noise components

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


%Figg=figure('Visible','off'); pause(0.1)

%imshow(a);

hold on        %first image output

pause(0.1)

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




%------------------------------ Setting the output image/s ------------------------------%

%message=
message1=sprintf('Total Baybayin words detected: %d\n (bounded by red boxes) \n\n\n Proceeding to transliterations, please wait.',row); %first output caption
end
