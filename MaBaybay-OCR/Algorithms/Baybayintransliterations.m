% ---------------------------------------------------------------------------------------%
% Transliteration of Detected Baybayin Words                                             %                           
% by Rodney Pino, Renier Mendoza, and Rachelle Sambayan                                  %
% Programmed by Rodney Pino at University of the Philippines - Diliman                   %
% Programming dates: September 2023 to January 2024                                      % 
% ---------------------------------------------------------------------------------------%


function [output2,transliterations2]=Baybayintransliterations(ModelUsed2,output,WBB,Baybayin_counter,Tagalog_words,A)
tic;
%ModelUsed2=Baybayin_Character_Classifier_00379;
%    t=timer('TimerFcn',@Baybayin_word_reader,'StartDelay',1);
% 
% %   t.TimerFcn = @(~,~)disp('3434324342343!');
%    t.ErrorFcnclc={@Baybayin_word_reader, 'Limittimeout'};
% start(t);
%while etime(clock, iniTime) < limit
 iniTime = clock;
 limit   = 150;  %transliteration limit time (in seconds)
%count=0; 
[row, ~]=size(WBB);
[row_out, ~]=find(output<0); row_out1=sum(row_out);
transliterations=cell(row_out1,1);
    parfor jj=1:row
           %  BOutP=find(output<1);
        if output(jj)~=1
           J=imcrop(A, WBB(jj,:)); 
           [transli,Idk]=Baybayin_word_reader(ModelUsed2, J,Tagalog_words,iniTime,limit);      %Transliterating each Baybayin word found
           
%            if etime(clock, iniTime) > limit
%                break;
%            end
           
%           count=1+count;
           if Idk==0 && length(transli)>3
               transli=transli(1:3,1);          %If the generated word is not in the dictionary, the system considers the first 3 Latin strings produced.
           end
           transliterations{jj,1}=transli;
        end
        
        
    end
       
%end
%stop(t); delete(t);

transliterations=transliterations(~cellfun('isempty',transliterations));  
transliterations2=transliterations;
toc;
  
     
%pad1=ceil(pad(2)/2);
% pad=size(A); pad2=uint8(ones(pad(1),365)*255); pad3=uint8(pad2); pad3(:,:,2)=pad3;
% pad4=cat(3,pad3,pad2); padresult=cat(2, A,pad4);

%padresult=insertText(padresult,[pad(2)+1 1], message, 'Fontsize',12,'BoxOpacity', 0,'Textcolor','black');       %inserting caption

if ~isempty(transliterations)
A4=A;   
for shift=1:row
    
   if output(shift)~=1
       
           
A1=WBB(shift,:);
A2_check=0;
 if A1(2)+A1(4) > size(A,1)
     if A1(1)+A1(3)>A(3)
        A2=A(A1(2):size(A,1),A1(1):A1(1)-1+A1(3),:); 
        
   A2_check=1;
    A3=255; A2(:)=A3;
        A4(A1(2):size(A,1),A1(1):A1(1)-1+A1(3),:)=A2;
        
     else
        A2=A(A1(2):size(A,1),A1(1):A1(1)+A1(3),:); 
    A2_check=1;
    A3=255; A2(:)=A3;
        A4(A1(2):size(A,1),A1(1):A1(1)+A1(3),:)=A2;
     end
    
        
 end
 
 if A1(1)+A1(3) > size(A,2)
     if A1(2)+A1(4)>A1(4)
        A2=A(A1(2):A1(2)-1+A1(4),A1(1):size(A,2),:);
        A2_check=1;
    A3=255; A2(:)=A3; 
A4(A1(2):A1(2)-1+A1(4),A1(1):size(A,2),:)=A2; 
     else
   A2=A(A1(2):A1(2)+A1(4),A1(1):size(A,2),:);
   A2_check=1;
   A3=255; A2(:)=A3; 
   A4(A1(2):A1(2)+A1(4),A1(1):size(A,2),:)=A2;
     end
 end
 
 if A2_check==0  
    if A1(2)+A1(4)>A1(4) 
    A2=A(A1(2):A1(2)-1+A1(4),A1(1):A1(1)+A1(3),:);
    A3=255; A2(:)=A3; 
A4(A1(2):A1(2)-1+A1(4),A1(1):A1(1)+A1(3),:)=A2;
    else
    A2=A(A1(2):A1(2)+A1(4),A1(1):A1(1)+A1(3),:);
    A3=255; A2(:)=A3; 
A4(A1(2):A1(2)+A1(4),A1(1):A1(1)+A1(3),:)=A2;        
    end
 end
 
   end
   
end

end


%text(17,11,yourtext,'Color','blue','FontUnits','Pixels','FontSize',20,'VerticalAlignment', 'top')
%padresult2=padresult; %orig padresult (with annotations)

%padresult=cat(2,padresult,A4);

if ~isempty(transliterations)
%messagenew=
fprintf('\nPossible Baybayin word/s transliteration are shown at the\nother image.');
%padresult2=insertText(padresult2,[pad(2)+1 48], messagenew, 'Fontsize',12,'BoxOpacity', 0,'Textcolor','black');  %inserting caption
end

 
 etime_check=etime(clock, iniTime);   
 if etime_check > limit
     error('Oops. System timeout for transliterations.');
 %    return;
 end
toc;    
    
% =============================================================== %
% Note:                                                           %
% If there are no Baybayin word detected, the system only outputs %
% the word script distinction image, where each recognized word   %
% is boxed by blue (for Latin words).                             %
% --------------------------------------------------------------- %    

%------------------- Second Image Output Setup -------------------%
if ~isempty(transliterations)

    kk=zeros(1,length(transliterations));

       single_line=cell(1,length(transliterations));
       
       
       
       for seq3=1:length(transliterations)
                if length(transliterations{seq3})>1
                    kk(seq3)=length(transliterations{seq3});
                    
                else
                    kk(seq3)=1;
                end
       end
       kk1=max(kk);
       
       for seq1=1:length(transliterations)
          
           k1=transliterations{seq1};
           if length(k1)==kk1
               single_line{seq1}=k1;
           else
              pad_k=kk1-length(k1);
              k1=cat(1,k1,cell(pad_k,1));  
              single_line{seq1}=k1;             
           end
       end
       
       single_lines=[];
        for seq4=1:length(single_line)
          single_lines=cat(2,single_lines,single_line{seq4});  
        end
         permn1=(1:kk1); permn2=permn(permn1,length(transliterations)); [permn2_row, permn2_col]=size(permn2);  
         
         single_lines2=cell(permn2_row, permn2_col);
         for set=1:permn2_row
             for set1=1:permn2_col
            single_lines2{set,set1}=single_lines{permn2(set,set1),set1};
             end
         end
     %    for set2=1:permn2_row
     %        discard_line=any(cellfun(@isempty, single_lines2(set2,:)), 2);
     %    end
     
     set3=zeros(1,permn2_row);
     for set2=1:permn2_row
        Check2=find(~cellfun(@isempty,single_lines2(set2,:)));
        if length(Check2)==length(transliterations)
            set3(set2)=1;
        end
     % single_lines3=single_lines2(~cellfun(@isempty, single_lines2(:,set2)), :);
     end
     nnz1=nnz(set3); single_lines3=cell(nnz1,length(transliterations)); kini=find(set3);
     
     for set4=1:length(kini)
     single_lines3(set4,:)=single_lines2(kini(set4),:);     % The system takes all the combinations of transliterated Baybayin words
                                                            % and are set to replace the Baybayin writings in the input image. 
     end
  

[line_row, ~]=size(single_lines3);

  % ============================================================
  % CONTEXT-AWARE DISAMBIGUATION 
  % Check if ANY word has multiple candidates (not just line_row > 1)
  % ============================================================
  has_ambiguous = false;
  for check_idx = 1:length(transliterations)
      if iscell(transliterations{check_idx}) && length(transliterations{check_idx}) > 1
          has_ambiguous = true;
          break;
      end
  end
  
  if has_ambiguous || line_row > 1
      fprintf('\n=== DISAMBIGUATION TRIGGERED ===\n');
      fprintf('line_row = %d, has_ambiguous = %d\n', line_row, has_ambiguous);
      try
          % Get number of words (columns) in original single_lines3
          [~, num_words] = size(single_lines3);
          fprintf('Calling disambiguate_candidates with %d words...\n', length(transliterations));
          disambiguated = disambiguate_candidates(transliterations);
          fprintf('Disambiguation returned: ');
          for dd = 1:length(disambiguated)
              fprintf('%s ', disambiguated{dd});
          end
          fprintf('\n');
          % Rebuild single_lines3 as 1 row x n columns (same structure as before)
          single_lines3 = cell(1, num_words);
          for di = 1:min(length(disambiguated), num_words)
              single_lines3{1, di} = disambiguated{di};
          end
          line_row = 1;  % Now we have only 1 row (the disambiguated result)
      catch ME
          fprintf('!!! Disambiguation FAILED: %s\n', char(ME.message));
          warning('Disambiguation failed: %s. Using first combination.', char(ME.message));
          single_lines3=single_lines3(1:1,:);   % Fallback to first combination
      end
  else
      fprintf('\n=== NO DISAMBIGUATION NEEDED (single candidates only) ===\n');
  end
  % ============================================================
    
%   if line_row>6
%       single_lines3=single_lines3(1:6,:);   % The system only takes at most 6 combinations of transliterated words for presentation purposes
%   end
  
  
[row_line, ~]=size(single_lines3);
sqrt_check=(mod(sqrt(row_line),1)==0);
  
figure('visible','off');
%Figg2=figure();
%imshow(padresult);

if row_line<4
    
    for sequence=1:row_line    
%        subplot(1,row_line,sequence);
%       imshow(A4);
        numseq=num2str(sequence);
        message3=['\underline{\bf Transliteration No. ',numseq,'}'];
        title(message3,'Fontsize', 15,'Interpreter','latex');
        transliterations=single_lines3(sequence,:);
        
        output2=A4; ff=cell(1,length(transliterations)); paramss=[]; %ff1=cell(1,length(transliterations));  paramss1=[];
        for trans=1:length(transliterations) 
        display_strings=transliterations{trans};

        
        ff{trans} = { @() text(WBB(Baybayin_counter(trans),1)+WBB(Baybayin_counter(trans),3)/2,WBB(Baybayin_counter(trans),2)+WBB(Baybayin_counter(trans),4)/2,display_strings),...
               @() rectangle('Position',WBB(Baybayin_counter(trans),:)) } ; 
           
        paramss=cat(2,paramss,{{{'Color','blue','FontUnits','Pixels','FontSize',0.69*WBB(Baybayin_counter(trans),4),'HorizontalAlignment','center'}, {'Edgecolor','r'}}});


        end



                 output2=insertInImage2(output2,ff,paramss);
    end

elseif row_line>3 && sqrt_check==1
   
    for sequence=1:row_line 
 %       subplot(sqrt(row_line),sqrt(row_line),sequence)
 %       imshow(A4);
        numseq=num2str(sequence);
        message3=['\underline{\bf Transliteration No. ',numseq,'}'];
        title(message3,'Fontsize', 15,'Interpreter','latex');
        transliterations=single_lines3(sequence,:);
        
        
        output2=A4; ff=cell(1,length(transliterations)); paramss=[]; %ff1=cell(1,length(transliterations));  paramss1=[];
        for trans=1:length(transliterations) 
        display_strings=transliterations{trans};


       ff{trans} = { @() text(WBB(Baybayin_counter(trans),1)+WBB(Baybayin_counter(trans),3)/2,WBB(Baybayin_counter(trans),2)+WBB(Baybayin_counter(trans),4)/2,display_strings),...
               @() rectangle('Position',WBB(Baybayin_counter(trans),:)) } ; 
           
        paramss=cat(2,paramss,{{{'Color','blue','FontUnits','Pixels','FontSize',0.69*WBB(Baybayin_counter(trans),4),'HorizontalAlignment','center'}, {'Edgecolor','r'}}});
        end 
        


                 output2=insertInImage2(output2,ff,paramss); 
            
    end
    
elseif row_line>3 && sqrt_check~=1
    
    check1=floor(sqrt(row_line));
    
    if row_line<=check1*(check1+1)
        
    for sequence=1:row_line 
%        subplot(check1,check1+1,sequence)
%        imshow(A4);
        numseq=num2str(sequence);
        message3=['\underline{\bf Transliteration No. ',numseq,'}'];
        title(message3,'Fontsize', 15,'Interpreter','latex');
        transliterations=single_lines3(sequence,:);
        
        output2=A4; ff=cell(1,length(transliterations)); paramss=[]; %ff1=cell(1,length(transliterations));  paramss1=[];        
        for trans=1:length(transliterations) 
        display_strings=transliterations{trans};


        ff{trans} = { @() text(WBB(Baybayin_counter(trans),1)+WBB(Baybayin_counter(trans),3)/2,WBB(Baybayin_counter(trans),2)+WBB(Baybayin_counter(trans),4)/2,display_strings),...
               @() rectangle('Position',WBB(Baybayin_counter(trans),:)) } ; 
           
        paramss=cat(2,paramss,{{{'Color','blue','FontUnits','Pixels','FontSize',0.69*WBB(Baybayin_counter(trans),4),'HorizontalAlignment','center'}, {'Edgecolor','r'}}});

        end
        


                 output2=insertInImage2(output2,ff,paramss);        
        
    end
    
    elseif row_line>check1*(check1+1) && row_line<(check1+1)*(check1+1)
        
    for sequence=1:row_line 
%        subplot(check1+1,check1+1,sequence)
%        imshow(A4);
        numseq=num2str(sequence);
        message3=['\underline{\bf Transliteration No. ',numseq,'}'];
        title(message3,'Fontsize', 15,'Interpreter','latex');
        transliterations=single_lines3(sequence,:);
        
        output2=A4; ff=cell(1,length(transliterations)); paramss=[]; %ff1=cell(1,length(transliterations));  paramss1=[];        
        for trans=1:length(transliterations) 
        display_strings=transliterations{trans};

        ff{trans} = { @() text(WBB(Baybayin_counter(trans),1)+WBB(Baybayin_counter(trans),3)/2,WBB(Baybayin_counter(trans),2)+WBB(Baybayin_counter(trans),4)/2,display_strings),...
               @() rectangle('Position',WBB(Baybayin_counter(trans),:)) } ; 
           
        paramss=cat(2,paramss,{{{'Color','blue','FontUnits','Pixels','FontSize',0.69*WBB(Baybayin_counter(trans),4),'HorizontalAlignment','center'}, {'Edgecolor','r'}}});
        end    
        

                 output2=insertInImage2(output2,ff,paramss);
        
    end
        
    end
    

end
  

end
%-----------------------------------------------------------------%

%----------------------------------------------------------------------------------------%

%hold off
toc; %tocBytes(gcp);