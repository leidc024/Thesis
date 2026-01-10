% ---------------------------------------------------------------------------------------%
% Baybayin OCR Package - OCR Type Selection                                              %                           
% by Rodney Pino, Renier Mendoza, and Rachelle Sambayan                                  %
% Programmed by Rodney Pino at University of the Philippines - Diliman                   %
% Programming dates: September 2023 to January 2024                                      % 
% ---------------------------------------------------------------------------------------%

%a=input text image
%OCRtype = select among three OCR type options. 
%Let OCRtype = 'c'  or OCRtype = 1 for Baybayin character OCR, OCRtype ='w' or OCRtype = 2 for Baybayin word OCR, and OCRtype = 'b' or OCRtype = 3 for Baybayin block OCR.')
function [output1,WBB,A,output,Baybayin_counter,message1]=BaybayinOCR(a,OCRtype,LvsB_classifier_00125)

if OCRtype==2
 [output1,WBB,A,output,Baybayin_counter,message1]=BaybayinxLatin_identifierP1(LvsB_classifier_00125,a);
 
elseif OCRtype==1 
 [output1,WBB,A,output,Baybayin_counter,message1]=Baybayin_identifier_onlyP1(a);   
else
    
 error('Oops. Something''s wrong.');
    
end