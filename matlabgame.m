% The Simple Game Engine is a class from object-oriented programming.
% If you are unfamiliar with object oriented programming, here is a quick
% crash course:
%
% Classes are a higher level of organizing programs beyond functions, they
% group together the functions (called methods) and variables (properties)
% of whatever it is you are trying to do. When you make a variable (called
% an object) from a class, it has all the properties from that class
% bundled together. This mimics how we naturally categorize things in real
% life. For example, cats are a class of animals, methods are the things a
% cat can do (e.g. pounce, meow, etc), properties describe a cat (e.g.
% color, age, location, etc), and objects  are individual cats (where each
% of the properties has a set value).
%
% The one extra bit of syntax you need to understand what's going on below
% is how to access properties of an object:
% Property "prop" of object "obj" is "obj.prop"

% The simpleGameEngine class inherets from the handle class because we
% want the game objects to be updated by their methods, specifically
% my_figure and my_image
classdef simpleGameEngine < handle
    properties
        sprites = {}; % color data of the sprites
        sprites_transparency = {}; % transparency data of the sprites
        sprite_width = 0;
        sprite_height = 0;
        background_color = [0, 0, 0];
        zoom = 1;
        my_figure; % figure identifier
        my_image;  % image data
    end
    
    methods
        function obj = simpleGameEngine(sprites_fname, sprite_height, sprite_width, zoom, background_color)
            % simpleGameEngine
            % Input: 
            %  1. File name of sprite sheet as a character array
            %  2. Height of the sprites in pixels
            %  3. Width of the sprites in pixels
            %  4. (Optional) Zoom factor to multiply image by in final figure (Default: 1)
            %  5. (Optional) Background color in RGB format as a 3 element vector (Default: [0,0,0] i.e. black)
            % Output: an SGE scene variable
            % Note: In RGB format, colors are specified as a mixture of red, green, and blue on a scale of 0 to 255. [0,0,0] is black, [255,255,255] is white, [255,0,0] is red, etc.
            % Example:
            %     	my_scene = simpleGameEngine('tictactoe.png',16,16,5,[0,150,0]);
            
            % load the input data into the object
            obj.sprite_width = sprite_width;
            obj.sprite_height = sprite_height;
            if nargin > 4
                obj.background_color = background_color;
            end
            if nargin > 3
                obj.zoom = zoom;
            end
            
            % read the sprites image data and transparency
            [sprites_image, ~, transparency] = imread(sprites_fname);
            
            % determine how many sprites there are based on the sprite size
            % and image size
            sprites_size = size(sprites_image);
            sprite_row_max = (sprites_size(1)+1)/(sprite_height+1);
            sprite_col_max = (sprites_size(2)+1)/(sprite_width+1);
            
            % Make a transparency layer if there is none (this happens when
            % there are no transparent pixels in the file).
            if isempty(transparency)
                transparency = 255*ones(sprites_size,'uint8');
            else
                % If there is a transparency layer, use repmat() to
                % replicate is to all three color channels
                transparency = repmat(transparency,1,1,3);
            end
            
            % loop over the image and load the individual sprite data into
            % the object
            for r=1:sprite_row_max
                for c=1:sprite_col_max
                    r_min = sprite_height*(r-1)+r;
                    r_max = sprite_height*r+r-1;
                    c_min = sprite_width*(c-1)+c;
                    c_max = sprite_width*c+c-1;
                    obj.sprites{end+1} = sprites_image(r_min:r_max,c_min:c_max,:);
                    obj.sprites_transparency{end+1} = transparency(r_min:r_max,c_min:c_max,:);
                end
            end
        end
        
        function drawScene(obj, background_sprites, foreground_sprites)
            % draw_scene 
            % Input: 
            %  1. an SGE scene, which gains focus
            %  2. A matrix of sprite IDs, the arrangement of the sprites in the figure will be the same as in this matrix
            %  3. (Optional) A second matrix of sprite IDs of the same size as the first. These sprites will be layered on top of the first set.
            % Output: None
            % Example: The following will create a figure with 3 rows and 3 columns of sprites
            %     	drawScene(my_scene, [4,5,6;7,8,9;10,11,12], [1,1,1;1,2,1;1,1,1]);
            
            scene_size = size(background_sprites);
            
            % Error checking: make sure the bg and fg are the same size
            if nargin > 2
                if ~isequal(scene_size, size(foreground_sprites))
                    error('Background and foreground matrices of scene must be the same size.')
                end
            end
            
            num_rows = scene_size(1);
            num_cols = scene_size(2);
            
            % initialize the scene_data array to the correct size and type
            scene_data = zeros(obj.sprite_height*num_rows, obj.sprite_width*num_cols, 3, 'uint8');
            
            % loop over the rows and colums of the tiles in the scene to
            % draw the sprites in the correct locations
            for tile_row=1:num_rows
                for tile_col=1:num_cols
                    
                    % Save the id of the current sprite(s) to make things
                    % easier to read later
                    bg_sprite_id = background_sprites(tile_row,tile_col);
                    if nargin > 2
                        fg_sprite_id = foreground_sprites(tile_row,tile_col);
                    end
                    
                    % Build the tile layer by layer, starting with the
                    % background color
                    tile_data = zeros(obj.sprite_height,obj.sprite_width,3,'uint8');
                    for rgb_idx = 1:3
                        tile_data(:,:,rgb_idx) = obj.background_color(rgb_idx);
                    end
                    
                    % Layer on the first sprite. Note that the tranparency
                    % data also ranges from 0 (transparent) to 255
                    % (visible)
                    tile_data = obj.sprites{bg_sprite_id} .* (obj.sprites_transparency{bg_sprite_id}/255) + ...
                        tile_data .* ((255-obj.sprites_transparency{bg_sprite_id})/255);
                    
                    % If needed, layer on the second sprite
                    if nargin > 2
                        tile_data = obj.sprites{fg_sprite_id} .* (obj.sprites_transparency{fg_sprite_id}/255) + ...
                            tile_data .* ((255-obj.sprites_transparency{fg_sprite_id})/255);
                    end
                    
                    % Calculate the pixel location of the top-left corner
                    % of the tile
                    rmin = obj.sprite_height*(tile_row-1);
                    cmin = obj.sprite_width*(tile_col-1);
                    
                    % Write the tile to the scene_data array
                    scene_data(rmin+1:rmin+obj.sprite_height,cmin+1:cmin+obj.sprite_width,:)=tile_data;
                end
            end
            
            % handle zooming
            big_scene_data = imresize(scene_data,obj.zoom,'nearest');
            
            % This part is a bit tricky, but avoids some latency, the idea
            % is that we only want to completely create a new figure if we
            % absolutely have to: the first time the figure is created,
            % when the old figure has been closed, or if the scene is
            % resized. Otherwise, we just update the image data in the
            % current image, which is much faster.
            if isempty(obj.my_figure) || ~isvalid(obj.my_figure)
                % inititalize figure
                obj.my_figure = figure();
                
                % set guidata to the  key press and release functions,
                % this allows keeping track of what key has been pressed
                obj.my_figure.KeyPressFcn = @(src,event)guidata(src,event.Key);
                obj.my_figure.KeyReleaseFcn = @(src,event)guidata(src,0);
                
                % actually display the image to the figure
                obj.my_image = imshow(big_scene_data,'InitialMagnification', 100);
                
            elseif isempty(obj.my_image)  || ~isprop(obj.my_image, 'CData') || ~isequal(size(big_scene_data), size(obj.my_image.CData))
                % Re-display the image if its size changed
                figure(obj.my_figure);
                obj.my_image = imshow(big_scene_data,'InitialMagnification', 100);
            else
                % otherwise just update the image data
                obj.my_image.CData = big_scene_data;
            end
        end
        
        function key = getKeyboardInput(obj)
            % getKeyboardInput
            % Input: an SGE scene, which gains focus
            % Output: next key pressed while scene has focus
            % Note: the operation of the program pauses while it waits for input
            % Example:
            %     	k = getKeyboardInput(my_scene);

            
            % Bring this scene to focus
            figure(obj.my_figure);
            
            % Pause the program until the user hits a key on the keyboard,
            % then return the key pressed. The loop is required so that
            % we don't exit on a mouse click instead.
            keydown = 0;
            while ~keydown
                keydown = waitforbuttonpress;
            end
            key = get(obj.my_figure,'CurrentKey');
        end
        
        function [row,col,button] = getMouseInput(obj)
            % getMouseInput
            % Input: an SGE scene, which gains focus
            % Output:
            %  1. The row of the tile clicked by the user
            %  2. The column of the tile clicked by the user
            %  3. (Optional) the button of the mouse used to click (1,2, or 3 for left, middle, and right, respectively)
            % 
            % Notes: A set of �crosshairs� appear in the scene�s figure,
            % and the program will pause until the user clicks on the
            % figure. It is possible to click outside the area of the
            % scene, in which case, the closest row and/or column is
            % returned.
            % 
            % Example:
            %     	[row,col,button] = getMouseInput (my_scene);
            
            % Bring this scene to focus
            figure(obj.my_figure);
            
            % Get the user mouse input
            [X,Y,button] = ginput(1);
            
            % Convert this into the tile row/column
            row = ceil(Y/obj.sprite_height/obj.zoom);
            col = ceil(X/obj.sprite_width/obj.zoom);
            
            % Calculate the maximum possible row and column from the
            % dimensions of the current scene
            sceneSize = size(obj.my_image.CData);
            max_row = sceneSize(1)/obj.sprite_height/obj.zoom;
            max_col = sceneSize(2)/obj.sprite_width/obj.zoom;
            
            % If the user clicked outside the scene, return instead the
            % closest row and/or column
            if row < 1
                row = 1;
            elseif row > max_row
                row = max_row;
            end
            if col < 1
                col = 1;
            elseif col > max_col
                col = max_col;
            end
        end
    end
end

clc
clear
close all
% Define variables used in RCade
framerate = 7.5; % frames per second
rtg_scene = simpleGameEngine('full_overworld_sprite_sheet_transparent.png', 16, 16, 5); % overworld sprite sheet
arcade_scene = simpleGameEngine('arcade_inside_sprite_sheet_transparent.png', 16, 16, 5); % arcade sprite sheet
shop_scene = simpleGameEngine('shop_inside_sprite_sheet_transparent.png', 16, 16, 5); % shop sprite sheet
lose_scene = simpleGameEngine('loser_sprite_sheet.png', 16, 16, 7); % losing sprite sheet
% Define matricies for background and foreground sprites
background = [1:27; 28:54; 55:81; 82:108; 109:135; 136:162; 163:189; 190:216; 217:243; 244:270; 271:297; 298:324];
backgroundArcade = [1:7; 8:14; 15:21; 22:28; 29:35; 36:42; 43:49];
backgroundShop = [1:9; 10:18; 19:27; 28:36; 37:45; 46:54; 55:63];
foreground = emptySprite * ones(12, 27);
foregroundArcade = emptySpriteArcade * ones(7, 7);
foregroundShop = emptySpriteShop * ones(7, 9);
% Define variables for displaying moneyValue in the top corner
emptySprite = 337;
displayCoin = 338;
displayZero = 339;
displayOne = 340;
displayTwo = 341;
displayThree = 342;
displayFour = 343;
displayFive = 344;
displaySix = 345;
displaySeven = 346;
displayEight = 347;
displayNine = 348;
displayGoldNine = 349;
displayCoinShop = 82;
displayZeroShop = 83;
displayOneShop = 84;
displayTwoShop = 85;
displayThreeShop = 86;
displayFourShop = 87;
displayFiveShop = 88;
displaySixShop = 91;
displaySevenShop = 92;
displayEightShop = 93;
displayNineShop = 94;
displayGoldNineShop = 95;
emptySpriteArcade = 62;
emptySpriteShop = 79;
xCoinCount = 1;
yCoinCountCoin = 24;
yCoinCountHundreds = 25;
yCoinCountTens = 26;
yCoinCountOnes = 27;
yCoinCountCoinShop = 6;
yCoinCountHundredsShop = 7;
yCoinCountTensShop = 8;
yCoinCountOnesShop = 9;
% Define movement variables for positions and sprites
xPos = 10;
yPos = 25;
xPosArcade = 6;
yPosArcade = 4;
xPosShop = 6;
yPosShop = 5;
charIdle = 325;
charIdleArcade = 50;
charIdleShop = 64;
charBikePurchased = 71;
bikeSpriteOne = 80;
bikeSpriteTwo = 81;
youWinOne = 89;
youWinTwo = 90;
% Display initial moneyValue and place character into overworld
foreground(xCoinCount, yCoinCountCoin) = displayCoin;
foreground(xCoinCount, yCoinCountHundreds) = displayOne;
foreground(xCoinCount, yCoinCountTens) = displayZero;
foreground(xCoinCount, yCoinCountOnes) = displayZero;
foreground(xPos, yPos) = charIdle;
foregroundArcade(xPosArcade, yPosArcade) = charIdleArcade;
foregroundShop(xCoinCount, yCoinCountCoinShop) = displayCoinShop;
foregroundShop(xPosShop, yPosShop) = charIdleShop;
drawScene(rtg_scene, background, foreground)
% Establish variables for movement cycles           // FIXME
walkCycle = 6; % global variable for the time between each walking sprite change
moveDown = 0;
moveUp = 0;
moveLeft = 0;
moveRight = 0;
moveDownArcade = 0;
moveUpArcade = 0;
moveLeftArcade = 0;
moveRightArcade = 0;
moveUpShop = 0;
moveLeftShop = 0;
moveDownShop = 0;
moveRightShop = 0;
% Define the initial money count and dummy variables for functions
moneyTotal = 100;
dummyInArg = 0;
dummyOutArg = 0;
moneyTotalChar = '';
while 1 % main game loop 
    tic
    rtg_scene.my_figure.KeyPressFcn = @(src,event)guidata(src,event.Key);
    % rtg_scene.my_figure.KeyReleaseFcn = @(src,event)guidata(src,'0');
    key_down = guidata(rtg_scene.my_figure); % user input
    % Use key_down to determine a sprite ID 
    if key_down
        sprite = str2double(key_down); 
    else
        sprite = 0;
    end
    
    % if it is a valid sprite ID (between 1 and 6), display it 
    if sprite >= 1 && sprite <= 6
        drawScene(rtg_scene,[sprite]) 
%     else
%         drawScene(rtg_scene, [1, 4, 7, 4, 7; 2, 5, 8, 5, 8; 3, 6, 9, 6, 9], [13, 10, 10, 10, 10; 10, 10, 10, 10, 10; 10, 10, 10, 10, 10])
    end
    
    % if statement that runs when player moves down
    if key_down == 's' & xPos < 12
        moveDown = moveDown + 1;
        oldxPos = xPos;
        xPos = xPos + 1;
        foreground(oldxPos, yPos) = emptySprite;
        foreground(xPos, yPos) = charIdle;
        drawScene(rtg_scene, background, foreground)
    end
    
    % if statement that runs when player moves up
    if key_down == 'w' & xPos > 1
        moveUp = moveUp + 1;
        oldxPos = xPos;
        xPos = xPos - 1;
        foreground(oldxPos, yPos) = emptySprite;
        foreground(xPos, yPos) = charIdle; 
        drawScene(rtg_scene, background, foreground)
    end
    
    % if statement that runs when player moves left
    if key_down == 'a' & yPos > 1
        moveLeft = moveLeft + 1;
        oldyPos = yPos;
        yPos = yPos - 1;
        foreground(xPos, oldyPos) = emptySprite;
        foreground(xPos, yPos) = charIdle;
        drawScene(rtg_scene, background, foreground)
    end
    
    % if statement that runs when player moves right
    if key_down == 'd' & yPos < 27
        moveRight = moveRight + 1;
        oldyPos = yPos;
        yPos = yPos + 1;
        foreground(xPos, oldyPos) = emptySprite;
        foreground(xPos, yPos) = charIdle;
        drawScene(rtg_scene, background, foreground)
    end
    
    % if statement that runs when player enters shop
    if xPos == 9 && yPos == 17
       oldxPos = xPos;
       xPos = 10;
       foreground(oldxPos, yPos) = emptySprite;
       foreground(xPos, yPos) = charIdle;
       close
       drawScene(shop_scene, backgroundShop, foregroundShop)
       while 1 % shop loop 
            tic
            shop_scene.my_figure.KeyPressFcn = @(src,event)guidata(src,event.Key);
            % rtg_scene.my_figure.KeyReleaseFcn = @(src,event)guidata(src,'0');
            key_down = guidata(shop_scene.my_figure); % user input
            % Use key_down to determine a sprite ID 
            if key_down
                sprite = str2double(key_down); 
            else
                sprite = 0;
            end
    
            % if it is a valid sprite ID (between 1 and 6), display it 
            if sprite >= 1 && sprite <= 6
                drawScene(shop_scene,[sprite]) 
%           else
%               drawScene(rtg_scene, [1, 4, 7, 4, 7; 2, 5, 8, 5, 8; 3, 6, 9, 6, 9], [13, 10, 10, 10, 10; 10, 10, 10, 10, 10; 10, 10, 10, 10, 10])
            end
    
            if key_down == 's' & xPosShop < 7
                moveDownShop = moveDownShop + 1;
                oldxPos = xPosShop;
                xPosShop = xPosShop + 1;
                foregroundShop(oldxPos, yPosShop) = emptySpriteShop;
                foregroundShop(xPosShop, yPosShop) = charIdleShop;
                drawScene(shop_scene, backgroundShop, foregroundShop)
            end
    
            if key_down == 'w' & xPosShop > 1
                moveUpShop = moveUpShop + 1;
                oldxPos = xPosShop;
                xPosShop = xPosShop - 1;
                foregroundShop(oldxPos, yPosShop) = emptySpriteShop;
                foregroundShop(xPosShop, yPosShop) = charIdleShop; 
                drawScene(shop_scene, backgroundShop, foregroundShop)
            end
    
            if key_down == 'a' & yPosShop > 1
                moveLeftShop = moveLeftShop + 1;
                oldyPos = yPosShop;
                yPosShop = yPosShop - 1;
                foregroundShop(xPosShop, oldyPos) = emptySpriteShop;
                foregroundShop(xPosShop, yPosShop) = charIdleShop;
                drawScene(shop_scene, backgroundShop, foregroundShop)
            end
    
            if key_down == 'd' & yPosShop < 9
                moveRightShop = moveRightShop + 1;
                oldyPos = yPosShop;
                yPosShop = yPosShop + 1;
                foregroundShop(xPosShop, oldyPos) = emptySpriteShop;
                foregroundShop(xPosShop, yPosShop) = charIdleShop;
                drawScene(shop_scene, backgroundShop, foregroundShop)
            end
    
            % if statement that runs when player leaves shop
            if xPosShop == 7 && yPosShop == 5
                clc;
                oldxPos = xPosShop;
                xPosShop = 6;
                foregroundShop(oldxPos, yPosShop) = emptySpriteShop;
                foregroundShop(xPosShop, yPosShop) = charIdleShop;
                close
                drawScene(rtg_scene, background, foreground)
                break
            end
            
            % if statement that runs when player interacts with cashier
            if xPosShop == 5 && yPosShop == 5
                oldxPos = xPosShop;
                xPosShop = 6;
                foregroundShop(oldxPos, yPosShop) = emptySpriteShop;
                foregroundShop(xPosShop, yPosShop) = charIdleShop;
                fprintf('\nAre you here for the bike? You came on a great day! We have this bike on sale for $999, on sale from $1000!');
                userInput = input('\nDo you wanna buy it? (Press "y" for yes, or press "n" for no.) ', 's');
                % user intends to purchase bike
                if userInput == 'y'
                    % user has enough money for bike
                    if moneyTotal >= 999
                        fprintf('\nOh, my... I didn''t expect anyone to actually have the money for this bike, much less actually buy it...');
                        fprintf('\nBut hey, if you want it so bad... it''s yours, kid.');
                        bikePurchase = -999;
                        moneyTotal = moneyTotal + bikePurchase;
                        displayTotalCoins
                        foregroundShop(xPosShop, yPosShop) = charBikePurchased;
                        foregroundShop(5, 5) = bikeSpriteOne;
                        foregroundShop(5, 6) = bikeSpriteTwo;
                        foregroundShop(6, 7) = youWinOne;
                        foregroundShop(6, 8) = youWinTwo;
                        drawScene(shop_scene, backgroundShop, foregroundShop)
                        userInput = input('\n\n You bought the bike, and you win! Thank you for playing RCade: The Demo! Press "q" to quit. ', 's');
                        % returns and terminates the program, player wins
                        if userInput == 'q'
                            return
                        else
                            % while loop to ensure user enters 'q'
                            while userInput ~= 'q'
                                userInput = input('', 's')
                            end
                            return
                        end
                    else
                        % player does not have enough money for bike
                        fprintf('\nYou don''t have enough money for the bike... did you expect me to give it to you for free?');
                        fprintf('\nCome back after you have enough money. It''s not like we get a whole lot of customers anyway...');
                    end
                else
                    % player is just looking for a new friend
                    fprintf('\nSo... what are you even here for? Just for friendly conversation? I''d love to talk, but I don''t get paid to chat.');
                end
            end
            
            % exits the while loop, mostly for debugging purposes
            if key_down == 'p'
                break
            end
    
            pause(1/framerate-toc); % wait for next frame
        end
    end
    
    % if statement that runs when player enters arcade
    if xPos == 5 && yPos == 6
        oldxPos = xPos;
        xPos = 6;
        foreground(oldxPos, yPos) = emptySprite;
        foreground(xPos, yPos) = charIdle;
        close
        drawScene(arcade_scene, backgroundArcade, foregroundArcade)
        while 1 % arcade loop 
            tic
            arcade_scene.my_figure.KeyPressFcn = @(src,event)guidata(src,event.Key);
            % rtg_scene.my_figure.KeyReleaseFcn = @(src,event)guidata(src,'0');
            key_down = guidata(arcade_scene.my_figure); % user input
            % Use key_down to determine a sprite ID 
            if key_down
                sprite = str2double(key_down); 
            else
                sprite = 0;
            end
    
            % if it is a valid sprite ID (between 1 and 6), display it 
            if sprite >= 1 && sprite <= 6
                drawScene(arcade_scene,[sprite]) 
%           else
%               drawScene(rtg_scene, [1, 4, 7, 4, 7; 2, 5, 8, 5, 8; 3, 6, 9, 6, 9], [13, 10, 10, 10, 10; 10, 10, 10, 10, 10; 10, 10, 10, 10, 10])
            end
    
            if key_down == 's' & xPosArcade < 7
                moveDownArcade = moveDownArcade + 1;
                oldxPos = xPosArcade;
                xPosArcade = xPosArcade + 1;
                foregroundArcade(oldxPos, yPosArcade) = emptySpriteArcade;
                foregroundArcade(xPosArcade, yPosArcade) = charIdleArcade;
                drawScene(arcade_scene, backgroundArcade, foregroundArcade)
            end
    
            if key_down == 'w' & xPosArcade > 1
                moveUpArcade = moveUpArcade + 1;
                oldxPos = xPosArcade;
                xPosArcade = xPosArcade - 1;
                foregroundArcade(oldxPos, yPosArcade) = emptySpriteArcade;
                foregroundArcade(xPosArcade, yPosArcade) = charIdleArcade; 
                drawScene(arcade_scene, backgroundArcade, foregroundArcade)
            end
    
            if key_down == 'a' & yPosArcade > 1
                moveLeftArcade = moveLeftArcade + 1;
                oldyPos = yPosArcade;
                yPosArcade = yPosArcade - 1;
                foregroundArcade(xPosArcade, oldyPos) = emptySpriteArcade;
                foregroundArcade(xPosArcade, yPosArcade) = charIdleArcade;
                drawScene(arcade_scene, backgroundArcade, foregroundArcade)
            end
    
            if key_down == 'd' & yPosArcade < 7
                moveRightArcade = moveRightArcade + 1;
                oldyPos = yPosArcade;
                yPosArcade = yPosArcade + 1;
                foregroundArcade(xPosArcade, oldyPos) = emptySpriteArcade;
                foregroundArcade(xPosArcade, yPosArcade) = charIdleArcade;
                drawScene(arcade_scene, backgroundArcade, foregroundArcade)
            end
    
            % if statement that runs when player exits arcade
            if xPosArcade == 7 && yPosArcade == 4
                oldxPos = xPosArcade;
                xPosArcade = 6;
                foregroundArcade(oldxPos, yPosArcade) = emptySpriteArcade;
                foregroundArcade(xPosArcade, yPosArcade) = charIdleArcade;
                close
                drawScene(rtg_scene, background, foreground)
                break
            end
            
            % if statement that runs when player approaches hangman machine
            if xPosArcade == 5 && yPosArcade == 1
                oldxPos = xPosArcade;
                xPosArcade = 6;
                foregroundArcade(oldxPos, yPosArcade) = emptySpriteArcade;
                foregroundArcade(xPosArcade, yPosArcade) = charIdleArcade;
                userInput = input('Would you like to play Hangman? Press "y" for yes, or press "n" for no. ', 's');
                % player wants to play hangman
                if userInput == 'y'
                    userInput = '';
                    pause;
                    % hangman function that runs the game code, takes
                    % in moneyTotal and returns moneyDifference
                    [moneyDifference, dummyOutArg] = hangmanGame(moneyTotal, dummyInArg);
                    % money earned/lost from hangman is added to the
                    % moneyTotal
                    moneyTotal = moneyTotal + moneyDifference;
                    % Class that changes the total number of coins held by
                    % the player
                    displayTotalCoins
                    playHangmanAgain = input('\nWould you like to play Hangman again? Press "y" for yes, or press "n" for no.', 's');
                    close
                    % while loop for if player wants to play hangman more
                    % than once
                    while playHangmanAgain == 'y'
                        userInput = '';
                        % hangman function that runs the game code, takes
                        % in moneyTotal and returns moneyDifference
                        [moneyDifference, dummyOutArg] = hangmanGame(moneyTotal, dummyInArg);
                        % money earned/lost from hangman is added to the
                        % moneyTotal
                        moneyTotal = moneyTotal + moneyDifference;
                        % Class that changes the total number of coins held by
                        % the player
                        displayTotalCoins
                        playHangmanAgain = input('\nWould you like to play Hangman again? Press "y" for yes, or press "n" for no.', 's');
                        close
                    end
                % player does not want to play hangman
                elseif userInput == 'n'
                    userInput = '';
                    oldxPos = xPosArcade;
                    xPosArcade = 6;
                    foregroundArcade(oldxPos, yPosArcade) = emptySpriteArcade;
                    foregroundArcade(xPosArcade, yPosArcade) = charIdleArcade;
                    drawScene(arcade_scene, backgroundArcade, foregroundArcade)
                end
            end
            
            % if statement that runs when player approaches o/u seven
            if xPosArcade == 5 && yPosArcade == 7
                oldxPos = xPosArcade;
                xPosArcade = 6;
                foregroundArcade(oldxPos, yPosArcade) = emptySpriteArcade;
                foregroundArcade(xPosArcade, yPosArcade) = charIdleArcade;
                userInput = input('Would you like to play Over/Under Seven? Press "y" for yes, or press "n" for no. ', 's');
                if userInput == 'y'
                    userInput = '';
                    % over under seven function that runs the game code, 
                    % takes in moneyTotal and returns moneyDifference
                    [moneyDifference, dummyOutArg] = overUnderSevenGame(moneyTotal, dummyInArg);
                    % money earned/lost from o/u seven is added to the
                    % moneyTotal
                    moneyTotal = moneyTotal + moneyDifference;
                    % Class that changes the total number of coins held by
                    % the player
                    displayTotalCoins
                    clc;
                elseif userInput == 'n'
                    userInput = '';
                    oldxPos = xPosArcade;
                    xPosArcade = 6;
                    foregroundArcade(oldxPos, yPosArcade) = emptySpriteArcade;
                    foregroundArcade(xPosArcade, yPosArcade) = charIdleArcade;
                    drawScene(arcade_scene, backgroundArcade, foregroundArcade)
                end
            end
            
            % if statement that runs when player approaches blackjack
            % dealer
            if xPosArcade == 3 && yPosArcade == 4
                oldxPos = xPosArcade;
                xPosArcade = xPosArcade + 1;
                foregroundArcade(oldxPos, yPosArcade) = emptySpriteArcade;
                foregroundArcade(xPosArcade, yPosArcade) = charIdleArcade;
                drawScene(arcade_scene, backgroundArcade, foregroundArcade)
                userInput = input('\nSo, kid, you think you can handle a round of Blackjack? (Press "y" for yes, or press "n" for no.) ', 's');
                % player wants to play blackjack
                if userInput == 'y'
                    userInput = '';
                    % blackjack function that runs the game code, 
                    % takes in moneyTotal and returns moneyDifference
                    [moneyDifference] = blackjackGame(moneyTotal);
                    pause;
                    oldxPos = xPosArcade;
                    xPosArcade = xPosArcade + 1;
                    key_down = 's';
                    foregroundArcade(oldxPos, yPosArcade) = emptySpriteArcade;
                    foregroundArcade(xPosArcade, yPosArcade) = charIdleArcade;
                    drawScene(arcade_scene, backgroundArcade, foregroundArcade)
                    % money earned/lost from blackjack is added to the
                    % moneyTotal
                    moneyTotal = moneyTotal + moneyDifference;
                    % Class that changes the total number of coins held by
                    % the player
                    displayTotalCoins
                % player does not want to play blackjack
                elseif userInput == 'n'
                    clc;
                    userInput = '';
                    fprintf('\nThen scram!')
                    oldxPos = xPosArcade;
                    xPosArcade = xPosArcade + 1;
                    key_down = 's';
                    foregroundArcade(oldxPos, yPosArcade) = emptySpriteArcade;
                    foregroundArcade(xPosArcade, yPosArcade) = charIdleArcade;
                    drawScene(arcade_scene, backgroundArcade, foregroundArcade)
                % player is literally the coolest and GOATed
                elseif userInput == 'derge'
                    clc;
                    userInput = '';
                    fprintf('\nOh, my apologies... I didn''t know you were a child of PRESTIGE and HONOR! Please, take the jackpot. On the house.')
                    fprintf('\nYou have received +$999.')
                    dergeGOAT = 999;
                    % automatically receive $999 added to moneyTotal
                    moneyTotal = moneyTotal + dergeGOAT;
                    key_down = 's';
                    foregroundArcade(oldxPos, yPosArcade) = emptySpriteArcade;
                    foregroundArcade(xPosArcade, yPosArcade) = charIdleArcade;
                    % Class that changes the total number of coins held by
                    % the player
                    displayTotalCoins
                    drawScene(arcade_scene, backgroundArcade, foregroundArcade)
                end
            end
    
            % exits the while loop, mostly for debugging purposes
            if key_down == 'p'
                break
            end
    
            pause(1/framerate-toc); % wait for next frame
        end
    end
    
    if moneyTotal <= 0
       % exits and terminates the program, player lost
       close
       clc;
       fprintf('\nCongratuations! You managed to lose ALL of your money! Are you proud? To be in debt? (neither are we, but here we are...)') 
       fprintf('\n\nGAME OVER')
       drawScene(lose_scene, [1:3;4:6])
       return
    end
    
    % exits the while loop, mostly for debugging purposes
    if key_down == 'p'
        break
    end
    
    pause(1/framerate-toc); % wait for next frame
end

function [moneyDifference, dummyOutArg] = hangmanGame(moneyTotal, dummyInArg)
%HANGMANGAME Summary of this function goes here
% Hangman Code
clc

% Define the scene and simpleGameEngine
hangman_scene = simpleGameEngine('hangman_sprite_sheet.png', 16, 16, 5);

% Define variables for the sprite sheet
poleBottom = 1;
poleTop = 2;
soleRope = 3;
blankSpace = 4;
soleHead = 5;
headLeftArm = 6;
headBothArms = 7;
soleBody = 8;
bodyLeftArm = 9;
bodyBothArms = 10;
bodyLeftLeg = 11;
bodyBothLegs = 12;
victoryStance = 13;
winCondition = 0;
dummyOutArg = 0;
dummyInArg = 0;

% Print pening Statements
fprintf('Hangman\n')

% Create empty string for stored guesses
storedGuesses = '';

% Import word options
wordList = importdata("hangman_word_list.txt");

% Generate a random word
w=randi(206);
% Set word that the user will try to guess equal to the random word
guess_word = wordList(w);
    
% Transform guess word cell array into a character string
guess_word=char(guess_word);

% Specify the length of the guess word generated
l=length(guess_word);

% For loop to display underscores for the length of the word being guessed
for d=1:l
    % Set each element to an underscore and print
    target(d)='_';
end

% Display underscores & spaces for the guess word and the hangman structure
letterDisplay = 1;
fprintf('\n')
while letterDisplay <= length(target)
    fprintf(target(letterDisplay))
    fprintf(' ')
    letterDisplay = letterDisplay + 1;
end
letterDisplay = 1;

% Display initial pole
drawScene(hangman_scene, [poleTop soleRope; poleBottom blankSpace])

% Print the length of the word
fprintf('\nYour word is %i letters long.', l)

% Ask the user to input a letter
userInput = input('\nPlease guess one letter: ', 's');

% While loop to ensure player only guesses one letter
while length(userInput) > 1
   userInput = input('\nPlease guess ONE letter: ', 's');
end
% Create empty string to store the user's guess
userGuess = '';
% For loop to create user guess string
for numChar = 1:l
    % Concatenate single character to length of character array
    userGuess = strcat(userGuess, userInput);
end

% Define variables for the for loop that checks if the user input matches
% character array
numLives = 0;
totalMatches = 0;
finalMatches = 0;
wordInProcess = '';
% For loop to check if user input letter matches character array (For loop
% to set the variable that will be used in guessing to the underscores of
% the word)
for i = 1:l
    wordInProcess(i) = '_';
end

% While loop that runs until the number of lives runs out
while numLives <= 5
    % Initializes the total number of matches to 0
    totalMatches = 0;
    % For loop to ensure that the user does not guess a letter more than
    % once
    for i = 1:length(storedGuesses)
        % While loop to detect if the user guesses a letter they already
        % guessed
        while userInput == storedGuesses(i)
            % Print that the user should not enter the same letter twice
            userInput = input('Do not enter the same letter twice. Please enter a unique letter. ', 's');
            % While loop to ensure player only guesses one letter
            while length(userInput) > 1
                userInput = input('\nPlease guess ONE letter: ', 's');
            end
        end
    end
    % For loop to check if the guess is correct
    for numChar = 1:l 
        % If statement for when the guess is correct
        if userGuess(numChar) == guess_word(numChar)
            % If true, adds 1 to the total amount of matches
            totalMatches = totalMatches + 1;
        end
    end
    % Calculate final matches as the sum of final matches and total matches
    finalMatches = finalMatches + totalMatches;
    
    % If statement to see if final matches is equal to the number of letters
    % in the word
    if finalMatches == l
        % If it's true, the player wins and the while loop is exited
        winCondition = 1;
        break
    end
    
    % If statement that runs when a match is made
    if totalMatches > 0
        % Print that the guess is correct
        fprintf('\nCorrect!')
        % Add guess to stored guesses
        storedGuesses = strcat(storedGuesses, userInput);
        % Print stored guesses
        fprintf('\nYour previous guesses include: ')
        % For loop that continuously displays the previous guesses
        for guessOutput = 1:length(storedGuesses)
            fprintf(storedGuesses(guessOutput))
        end
        
        % Start a new line
        fprintf('\n')
        
        % For loop that displays the correct guess in place of an underscore
        for i = 1:l
            % If statement that tests if the guess is correct
            if userGuess(i) == guess_word(i)
                % Sets the guess equal to the correct letter
                wordInProcess(i) = guess_word(i);
                % Display the letter
                fprintf(wordInProcess(i))
                fprintf(' ')
            % Run if the guess is incorrect
            else
                % Keep the underscore
                wordInProcess(i);
                % Display the underscore
                fprintf(wordInProcess(i))
                fprintf(' ')
            end
        end
        
        % Ask the user to input a letter
        userInput = input('\nPlease guess one letter: ', 's');
        
        % While loop to ensure player only guesses one letter
        while length(userInput) > 1
            userInput = input('\nPlease guess ONE letter: ', 's');
        end
        
        userGuess = '';
        % For loop to create user guess string
        for numChar = 1:l
            % Concatenate single character to length of character array
            userGuess = strcat(userGuess, userInput);
        end
    % Else if statement that runs when the guess is incorrect
    elseif totalMatches == 0
        % Print that the guess is incorrect
        fprintf('\nIncorrect.')
        
        % Add incorrect guess to stored guesses
        storedGuesses = strcat(storedGuesses, userInput);
        % Print stored guesses
        fprintf('\nYour previous guesses include: ')
        % For loop that continuously displays stored guesses
        for guessOutput = 1:length(storedGuesses)
            fprintf(storedGuesses(guessOutput))
        end
        
        % Start a new line
        fprintf('\n')
        
        % Add to the number of lives you've used
        numLives = numLives + 1;
        
        % If statement for when 1 life has been used
        if numLives == 1;
            % Display pole and head
            drawScene(hangman_scene, [poleTop soleHead; poleBottom blankSpace])
        % Else if statement for when 2 lives have been used
        elseif numLives == 2;
            % Display the torso
            drawScene(hangman_scene, [poleTop soleHead; poleBottom soleBody])
        % Else if statement for when 3 lives have been used
        elseif numLives == 3;
            % Display left arm
            drawScene(hangman_scene, [poleTop headLeftArm; poleBottom bodyLeftArm])
        % Else if statement for when 4 lives have been used
        elseif numLives == 4;
            % Display right arm
            drawScene(hangman_scene, [poleTop headBothArms; poleBottom bodyBothArms])
        % Else if statement for when 5 lives have been used
        elseif numLives == 5;
            % Display left leg
            drawScene(hangman_scene, [poleTop headBothArms; poleBottom bodyLeftLeg])
        end
        
        % For loop that displays the correct guess in place of an underscore
        for i = 1:l
            % If statement for when the guess is correct
            if userGuess(i) == guess_word(i)
                % Set the guess equal to the correct letter
                wordInProcess(i) = guess_word(i);
                % Display the letter
                fprintf(wordInProcess(i))
                fprintf(' ')
            % Run if the guess is incorrect
            else
                % Keep the underscore
                wordInProcess(i);
                % Display the underscore
                fprintf(wordInProcess(i))
                fprintf(' ')
            end
        end
        
        %Ask the user to input a letter
        userInput = input('\nPlease guess one letter: ', 's');
        
        % While loop to ensure player only guesses one letter
        while length(userInput) > 1
            userInput = input('\nPlease guess ONE letter: ', 's');
        end
        
        %Create empty string 
        userGuess = '';
        
        % For loop to create user guess string
        for numChar = 1:l
            % Concatenate single character to length of character array
            userGuess = strcat(userGuess, userInput);
        end
        
    end
end
% Display if the user lost or won
    % If statement that runs if the user lost
    if winCondition == 0
        % Display the the full body hanging
        drawScene(hangman_scene, [poleTop headBothArms; poleBottom bodyBothLegs])
        % Print that the user lost and the correct word
        fprintf('\nSorry, you lost. The correct word was "')
        fprintf(guess_word)
        fprintf('."')
        moneyDifference = -20;
    % Else if statement that runs if the user won
    elseif winCondition == 1
        % Display the body jumping out of the noose
        drawScene(hangman_scene, [poleTop soleRope; poleBottom victoryStance])
        % Print that the user won and the correct word
        fprintf('\nCongratulations! You win! The word was "')
        fprintf(guess_word)
        fprintf('."')
        moneyDifference = 20;
    end
end

function [moneyDifference, dummyOutArg] = overUnderSevenGame(moneyTotal, dummyInArg)
%HIGH LOW SEVENS CODE
clc

ouseven_scene = simpleGameEngine('over_under_seven_sprite_sheet.png', 16, 16, 5);
drawScene(ouseven_scene, [13:14])
win = [15:16];
lose = [17:18];
%Creates the first dice
dice1 = randi([0,6], 1, 1);
%Creates the second dice
dice2 = randi([0,6], 1, 1);
%Sums the total of the dice 
dicesum = dice1+dice2;

dummyInArg = 0;
dummyOutArg = 0;
moneyDifference = 0;

playagain = 1;
while playagain > 0
    
%Creates the first dice
dice1 = randi([0,6], 1, 1);
%Creates the second dice
dice2 = randi([0,6], 1, 1);
%Sums the total of the dice 
dicesum = dice1+dice2;

%Takes in initial bet for over, under, or equal to seven.
drawScene(ouseven_scene, [13:14])
bet=input('Is dice sum greater than, less than, or equal to seven? Input 10 for greater, 1 for less than, and 7 for equal to.\n');

                %Checks to see if bet was inputted with the correct number
                if (bet == 10 || bet == 1 || bet == 7)
                    
                        %If bet is greater and the dicesum is over 7 
                        if (bet == 10 && dicesum > 7) 
                            fprintf('Dice sum was %i. ', dicesum)
                            fprintf('You win!\n');
                            moneyDifference = moneyDifference + 5;
                            drawScene(ouseven_scene, [dice1, dice2 + 6; win])
                           
                           %If bet is greater and dicesum is under 7
                        elseif (bet == 10 && dicesum < 7)
                            fprintf('Dice sum was %i. ', dicesum)
                            fprintf('You lose :( \n');
                            moneyDifference = moneyDifference - 5;
                            drawScene(ouseven_scene, [dice1, dice2 + 6; lose])
                            
                        %If bet is less than 7 and dice sum is over 7
                        elseif (bet == 1 && dicesum > 7) 
                            fprintf('Dice sum was %i. ', dicesum)
                            fprintf('You lose :( \n'); 
                            moneyDifference = moneyDifference - 5;
                            drawScene(ouseven_scene, [dice1, dice2 + 6; lose])
                        
                        %If bet is less than 7 and dice sum is under 7    
                        elseif (bet == 1 && dicesum < 7)
                            fprintf('Dice sum was %i. ', dicesum)
                            fprintf('You win!\n');
                            moneyDifference = moneyDifference + 5;
                            drawScene(ouseven_scene, [dice1, dice2 + 6; win])
                        
                        %If bet is equal to and dice sum is equal to 7    
                        elseif (bet == 7 && dicesum == 7)
                            fprintf('Dice sum was %i. ', dicesum)
                            fprintf('You win!\n');
                            moneyDifference = moneyDifference + 5;
                            drawScene(ouseven_scene, [dice1, dice2 + 6; win])
                        
                        %If bet is equal to and the dicesum is not 7    
                        else
                            fprintf('Dice sum was %i. ', dicesum)
                            fprintf('You lose :( \n') ;
                            moneyDifference = moneyDifference - 5;
                            drawScene(ouseven_scene, [dice1, dice2 + 6; lose])
                            
                        end
                        
                %Has user input new bet to compare to the dice sum         
                else 
                    fprintf('Incorrect input. Please try again \n')
                    bet=input('Is dice sum greater than, less than, or equal to seven? Input 10 for greater, 1 for less than, and 7 for equal to.\n');
                end

                playagain=input('Play again? If yes, input 1. If no, input 0');
end

%Ends the game. Ensures the user has a wonderful day.                
fprintf('Thank you for playing. Have a wonderful day.')
close
end

function [moneyDifference] = blackjackGame(moneyTotal)
% %Class where the game is started.
%sets the starting money value, then starts the game code by calling class
%playblackjack
moneyvalue=moneyTotal;
playblackjack
moneyDifference = moneyvalue - moneyTotal;
end

clc;
%starts the true gameplay

    %runs card assignments of player and dealer through giant if statement in a "BlackJackDecisions" class
    % made to assign names and values to the numbers given from the
    % randperm() function
    BlackJackDecisions
    %card names and values assigned to each card, displays the results in
    %command window.
    clc;
    fprintf('The Dealer shows: %s\n\n',dealerCard1Name)
    fprintf('You have: %s and %s\n',playerCard1Name,playerCard2Name)
    fprintf('Your total is %d\n\n',playercount)
    fprintf('Press any key to continue.')
    pause;
    
    %player decides what to do for his/her turn through this class
    PlayerDecisions
    
    %conducts AI moves for dealer through this class
    DealerDecisions
    
%determines results if dealer has less than 21 and player has less than 21
%if there is no blackjack or bust apparent, then this class checks counts 
%for both to see which total is closest to 21

%first if is to see if there is a tie, if tied, the house always wins (the
%dealer).
if playercount==dealercount
    clc;
    %Displays dealer winning screen for a tie
    fprintf('Dealer has: %d\n',dealercount)
    fprintf('You have: %d\n',playercount)
    fprintf('Dealer says: "We tied! The House always wins on a tie!"\n\n')
    fprintf('You lose your bet.\n')
    %Doesn't add money back that was taken, bet money is lost by user.
    fprintf('You now have %0.2f dollars.\n',moneyvalue)
    fprintf('Press any key to continue')
    %Any key to continue concept using "pause", then resets back to the
    %beginning of the game at playblackjack class.
    pause;
    playblackjack
    
    %next if statement is for when the players value count is greater than
    %the dealers.
elseif playercount > dealercount
    clc;
    %Displays player winning screen
    fprintf('Dealer has: %d\n',dealercount)
    fprintf('You have: %d\n',playercount)
    fprintf('Dealer says: "You won!"\n\n')
    %Creates value for money that player won, adds that money value to
    %their total.
    betWin=(playerBet*2);
    moneyvalue=(moneyvalue+betWin);
    fprintf('The dealer gives you %0.2f dollars.\n',betWin)
    fprintf('You now have %0.2f dollars.\n\n',moneyvalue)
    fprintf('Press any key to return to continue.')
    %"pause" then "playblackjack" combination to reset the game back to the
    %start after a key is pressed.
    pause;
    playblackjack
    
 %final if statement is when the dealer has a greater count than the player.   
elseif playercount < dealercount
    clc;
    %Displays dealer winning screen
    %bet money not given back
    fprintf('Dealer has: %d\n',dealercount)
    fprintf('You have: %d\n',playercount)
    fprintf('Dealer says: "Sorry, but you lost."\n\n')
    fprintf('You have lost the money you bet.\n')
    fprintf('Press any key to continue.\n')
    %Pauses for "press any key" functionality, resets back to beginning.
    pause;
    playblackjack
end
%ended
    
%Class created for dealer hitting

% singular large if loop for the dealer hitting in blackjack from "dealblackjack"
%Checks the value given from randperm() function and establishes card name
%and card value from it. Once again very long, but works nonetheless.  
if dealercardadd==1
    dealercardaddName=('Ace of Spades');
    %determines if count for the Ace should use 11 or 1
    checkdealercard=((dealercount)+11);
    %Checks to see if dealer will bust
    if checkdealercard > 21 
        dealercardaddValue=1;
    else 
        dealercardaddValue=11;
    end
    
elseif dealercardadd==2
    dealercardaddName=('Ace of Clubs');
    %determines if count for the Ace should use 11 or 1
    checkdealercard=((dealercount)+11);
    %Checks to see if dealer will bust
    if checkdealercard > 21 
        dealercardaddValue=1;
    else 
        dealercardaddValue=11;
    end
    
elseif dealercardadd==3
    dealercardaddName=('Ace of Diamonds');
    %determines if count for the Ace should use 11 or 1
    checkdealercard=((dealercount)+11);
    %Checks to see if dealer will bust
    if checkdealercard > 21 
        dealercardaddValue=1;
    else 
        dealercardaddValue=11;
    end
    
elseif dealercardadd==4
    dealercardaddName=('Ace of Hearts');
    %determines if count for the Ace should use 11 or 1
    checkdealercard=((dealercount)+11);
    %Checks to see if dealer will bust
    if checkdealercard > 21 
        dealercardaddValue=1;
    else 
        dealercardaddValue=11;
    end
        %Rest of values are matched with a specific card of the 52-card deck
elseif dealercardadd==5
    dealercardaddName=('Two of Spades');
    dealercardaddValue=2;
    
elseif dealercardadd==6
    dealercardaddName=('Two of Clubs');
    dealercardaddValue=2;
    
elseif dealercardadd==7
    dealercardaddName=('Two of Diamonds');
    dealercardaddValue=2;
    
elseif dealercardadd==8
    dealercardaddName=('Two of Hearts');
    dealercardaddValue=2;
    
elseif dealercardadd==9
    dealercardaddName=('Three of Spades');
    dealercardaddValue=3;
    
elseif dealercardadd==10
    dealercardaddName=('Three of Clubs');
    dealercardaddValue=3;
    
elseif dealercardadd==11
    dealercardaddName=('Three of Diamonds');
    dealercardaddValue=3;
    
elseif dealercardadd==12
    dealercardaddName=('Three of Hearts');
    dealercardaddValue=3;
    
elseif dealercardadd==13
    dealercardaddName=('Four of Spades');
    dealercardaddValue=4;
    
elseif dealercardadd==14
    dealercardaddName=('Four of Clubs');
    dealercardaddValue=4;
    
elseif dealercardadd==15
    dealercardaddName=('Four of Diamonds');
    dealercardaddValue=4;
    
elseif dealercardadd==16
    dealercardaddName=('Four of Hearts');
    dealercardaddValue=4;
    
elseif dealercardadd==17
    dealercardaddName=('Five of Spades');
    dealercardaddValue=5;
    
elseif dealercardadd==18
    dealercardaddName=('Five of Clubs');
    dealercardaddValue=5;
    
elseif dealercardadd==19
    dealercardaddName=('Five of Diamonds');
    dealercardaddValue=5;
    
elseif dealercardadd==20
    dealercardaddName=('Five of Hearts');
    dealercardaddValue=5;
    
elseif dealercardadd==21
    dealercardaddName=('Six of Spades');
    dealercardaddValue=6;
    
elseif dealercardadd==22
    dealercardaddName=('Six of Clubs');
    dealercardaddValue=6;
    
elseif dealercardadd==23
    dealercardaddName=('Six of Diamonds');
    dealercardaddValue=6;
    
elseif dealercardadd==24
    dealercardaddName=('Six of Hearts');
    dealercardaddValue=6;
    
elseif dealercardadd==25
    dealercardaddName=('Seven of Spades');
    dealercardaddValue=7;
    
elseif dealercardadd==26
    dealercardaddName=('Seven of Clubs');
    dealercardaddValue=7;
    
elseif dealercardadd==27
    dealercardaddName=('Seven of Diamonds');
    dealercardaddValue=7;
    
elseif dealercardadd==28
    dealercardaddName=('Seven of Hearts');
    dealercardaddValue=7;
    
elseif dealercardadd==29
    dealercardaddName=('Eight of Spades');
    dealercardaddValue=8;
    
elseif dealercardadd==30
    dealercardaddName=('Eight of Clubs');
    dealercardaddValue=8;
    
elseif dealercardadd==31
    dealercardaddName=('Eight of Diamonds');
    dealercardaddValue=8;
    
elseif dealercardadd==32
    dealercardaddName=('Eight of Hearts');
    dealercardaddValue=8;
    
elseif dealercardadd==33
    dealercardaddName=('Nine of Spades');
    dealercardaddValue=9;
    
elseif dealercardadd==34
    dealercardaddName=('Nine of Clubs');
    dealercardaddValue=9;
    
elseif dealercardadd==35
    dealercardaddName=('Nine of Diamonds');
    dealercardaddValue=9;
    
elseif dealercardadd==36
    dealercardaddName=('Nine of Hearts');
    dealercardaddValue=9;
    
elseif dealercardadd==37
    dealercardaddName=('Ten of Spades');
    dealercardaddValue=10;
    
elseif dealercardadd==38
    dealercardaddName=('Ten of Clubs');
    dealercardaddValue=10;
    
elseif dealercardadd==39
    dealercardaddName=('Ten of Diamonds');
    dealercardaddValue=10;
    
elseif dealercardadd==40
    dealercardaddName=('Ten of Hearts');
    dealercardaddValue=10;
    
elseif dealercardadd==41
    dealercardaddName=('Jack of Spades');
    dealercardaddValue=10;
    
elseif dealercardadd==42
    dealercardaddName=('Jack of Clubs');
    dealercardaddValue=10;
    
elseif dealercardadd==43
    dealercardaddName=('Jack of Diamonds');
    dealercardaddValue=10;
    
elseif dealercardadd==44
    dealercardaddName=('Jack of Hearts');
    dealercardaddValue=10;
    
elseif dealercardadd==45
    dealercardaddName=('Queen of Spades');
    dealercardaddValue=10;
    
elseif dealercardadd==46
    dealercardaddName=('Queen of Clubs');
    dealercardaddValue=10;
    
elseif dealercardadd==47
    dealercardaddName=('Queen of Diamonds');
    dealercardaddValue=10;
    
elseif dealercardadd==48
    dealercardaddName=('Queen of Hearts');
    dealercardaddValue=10;
    
elseif dealercardadd==49
    dealercardaddName=('King of Spades');
    dealercardaddValue=10;
    
elseif dealercardadd==50
    dealercardaddName=('King of Clubs');
    dealercardaddValue=10;
    
elseif dealercardadd==51
    dealercardaddName=('King of Diamonds');
    dealercardaddValue=10;
    
elseif dealercardadd==52
    dealercardaddName=('King of Hearts');
    dealercardaddValue=10;
end

%class for the decisions for the dealer
%dealer hits on anything at a value of 16 or less 
%dealer stands on anything ata value 17 or higher
clc;
%Creates checkdealer variable, displays the cards and the value the dealer
%has.
checkdealer=0;
fprintf('The Dealer turns over a %s\n\n',dealerCard2Name)
fprintf('The Dealer has %s and %s\n\n',dealerCard1Name,dealerCard2Name)
fprintf('The Dealer''s total is %d\n\n',dealercount)
%press any key to continue process using pause
fprintf('Press any key to continue.')
pause;

%hits while count is less than or equal to 16, therefore while loop is
%established for hitting
while dealercount <= 16
    %Since there are four matrix locators, adding +4 for the addition of a
    %new card will create a brand new matrix locator for the next card.
    dealCard1=(dealCard1+4);
    %variable value to add to the total value of dealers cards.
    dealercardadd=(dealCard(dealCard1));
    %Calls the class for Dealer if they choose to hit, goes through hitting
    %process
    DealerHitting
    %Clears
    clc;
    %Displays what value and name of hit card was for dealer
    fprintf('The Dealer hits himself with %s\n\n',dealercardaddName)
    %new dealer count
    dealercount=(dealercount+dealercardaddValue);
    fprintf('The Dealer now has a total of %d\n\n',dealercount)
    %pause function
    fprintf('Press any key to continue.')
    pause;
end

%while loop ends when dealercount is greater than 16

%if statement to check for blackjack or bust from the dealer
if dealercount==21
    %in this scenario, the dealer has blackjack
    clc;
    %Displays dealer blackjack screen
    fprintf('Dealer says: "I''ve got Blackjack! Looks like you lose."\n\n')
    fprintf('You have lost your bet.\n')
    %bet money is not returned
    fprintf('You now have %0.2f dollars.\n',moneyvalue)
    %pauses, any key returns to beginning of game
    fprintf('Press any key to continue\n')
    pause;
    playblackjack
    
    
elseif dealercount > 21
    %in this scenario, the dealer has busted
    clc;
    %Displays dealer bust screen
    fprintf('Dealer: "Dealer busts on this hand. You win!"\n\n')
    %win value is created and added to total money value of player.
    betWin=(playerBet*2);
    moneyvalue=(moneyvalue+betWin);
    %displays what was won
    fprintf('The dealer gives you %0.2f dollars.\n',betWin)
    fprintf('You now have %0.2f dollars.\n\n',moneyvalue)
    fprintf('Press any key to return to continue.')
    %pauses, any key returns to beginning of game
    pause;
    playblackjack
else
    %the dealer stopped hitting and has less than 21 therefore it returns 
    %to the dealblackjack class from where it was called
    %to continue on for evaluation.
end
%returns to dealblackjack
    
%Class created for player hitting

% singular large if loop for the player hitting in blackjack from "dealblackjack"
%Checks the value given from randperm() function and establishes card name
%and card value from it. Once again very long, but works nonetheless.  
if playercardadd==1
    playercardaddName=('Ace of Spades');
    %determines if count for the Ace should use 11 or 1
    checkPlayerCard=((playercount)+11);
    %Checks to see if player will bust
    if checkPlayerCard > 21 
        playercardaddValue=1;
    else 
        playercardaddValue=11;
    end
    
elseif playercardadd==2
    playercardaddName=('Ace of Clubs');
    %determines if count for the Ace should use 11 or 1
    checkPlayerCard=((playercount)+11);
    %Checks to see if player will bust
    if checkPlayerCard > 21 
        playercardaddValue=1;
    else 
        playercardaddValue=11;
    end
    
elseif playercardadd==3
    playercardaddName=('Ace of Diamonds');
    %determines if count for the Ace should use 11 or 1
    checkPlayerCard=((playercount)+11);
    %Checks to see if player will bust
    if checkPlayerCard > 21 
        playercardaddValue=1;
    else 
        playercardaddValue=11;
    end
    
elseif playercardadd==4
    playercardaddName=('Ace of Hearts');
    %determines if count for the Ace should use 11 or 1
    checkPlayerCard=((playercount)+11);
    %Checks to see if player will bust
    if checkPlayerCard > 21 
        playercardaddValue=1;
    else 
        playercardaddValue=11;
    end
      %Rest of values are matched with a specific card of the 52-card deck  
elseif playercardadd==5
    playercardaddName=('Two of Spades');
    playercardaddValue=2;
    
elseif playercardadd==6
    playercardaddName=('Two of Clubs');
    playercardaddValue=2;
    
elseif playercardadd==7
    playercardaddName=('Two of Diamonds');
    playercardaddValue=2;
    
elseif playercardadd==8
    playercardaddName=('Two of Hearts');
    playercardaddValue=2;
    
elseif playercardadd==9
    playercardaddName=('Three of Spades');
    playercardaddValue=3;
    
elseif playercardadd==10
    playercardaddName=('Three of Clubs');
    playercardaddValue=3;
    
elseif playercardadd==11
    playercardaddName=('Three of Diamonds');
    playercardaddValue=3;
    
elseif playercardadd==12
    playercardaddName=('Three of Hearts');
    playercardaddValue=3;
    
elseif playercardadd==13
    playercardaddName=('Four of Spades');
    playercardaddValue=4;
    
elseif playercardadd==14
    playercardaddName=('Four of Clubs');
    playercardaddValue=4;
    
elseif playercardadd==15
    playercardaddName=('Four of Diamonds');
    playercardaddValue=4;
    
elseif playercardadd==16
    playercardaddName=('Four of Hearts');
    playercardaddValue=4;
    
elseif playercardadd==17
    playercardaddName=('Five of Spades');
    playercardaddValue=5;
    
elseif playercardadd==18
    playercardaddName=('Five of Clubs');
    playercardaddValue=5;
    
elseif playercardadd==19
    playercardaddName=('Five of Diamonds');
    playercardaddValue=5;
    
elseif playercardadd==20
    playercardaddName=('Five of Hearts');
    playercardaddValue=5;
    
elseif playercardadd==21
    playercardaddName=('Six of Spades');
    playercardaddValue=6;
    
elseif playercardadd==22
    playercardaddName=('Six of Clubs');
    playercardaddValue=6;
    
elseif playercardadd==23
    playercardaddName=('Six of Diamonds');
    playercardaddValue=6;
    
elseif playercardadd==24
    playercardaddName=('Six of Hearts');
    playercardaddValue=6;
    
elseif playercardadd==25
    playercardaddName=('Seven of Spades');
    playercardaddValue=7;
    
elseif playercardadd==26
    playercardaddName=('Seven of Clubs');
    playercardaddValue=7;
    
elseif playercardadd==27
    playercardaddName=('Seven of Diamonds');
    playercardaddValue=7;
    
elseif playercardadd==28
    playercardaddName=('Seven of Hearts');
    playercardaddValue=7;
    
elseif playercardadd==29
    playercardaddName=('Eight of Spades');
    playercardaddValue=8;
    
elseif playercardadd==30
    playercardaddName=('Eight of Clubs');
    playercardaddValue=8;
    
elseif playercardadd==31
    playercardaddName=('Eight of Diamonds');
    playercardaddValue=8;
    
elseif playercardadd==32
    playercardaddName=('Eight of Hearts');
    playercardaddValue=8;
    
elseif playercardadd==33
    playercardaddName=('Nine of Spades');
    playercardaddValue=9;
    
elseif playercardadd==34
    playercardaddName=('Nine of Clubs');
    playercardaddValue=9;
    
elseif playercardadd==35
    playercardaddName=('Nine of Diamonds');
    playercardaddValue=9;
    
elseif playercardadd==36
    playercardaddName=('Nine of Hearts');
    playercardaddValue=9;
    
elseif playercardadd==37
    playercardaddName=('Ten of Spades');
    playercardaddValue=10;
    
elseif playercardadd==38
    playercardaddName=('Ten of Clubs');
    playercardaddValue=10;
    
elseif playercardadd==39
    playercardaddName=('Ten of Diamonds');
    playercardaddValue=10;
    
elseif playercardadd==40
    playercardaddName=('Ten of Hearts');
    playercardaddValue=10;
    
elseif playercardadd==41
    playercardaddName=('Jack of Spades');
    playercardaddValue=10;
    
elseif playercardadd==42
    playercardaddName=('Jack of Clubs');
    playercardaddValue=10;
    
elseif playercardadd==43
    playercardaddName=('Jack of Diamonds');
    playercardaddValue=10;
    
elseif playercardadd==44
    playercardaddName=('Jack of Hearts');
    playercardaddValue=10;
    
elseif playercardadd==45
    playercardaddName=('Queen of Spades');
    playercardaddValue=10;
    
elseif playercardadd==46
    playercardaddName=('Queen of Clubs');
    playercardaddValue=10;
    
elseif playercardadd==47
    playercardaddName=('Queen of Diamonds');
    playercardaddValue=10;
    
elseif playercardadd==48
    playercardaddName=('Queen of Hearts');
    playercardaddValue=10;
    
elseif playercardadd==49
    playercardaddName=('King of Spades');
    playercardaddValue=10;
    
elseif playercardadd==50
    playercardaddName=('King of Clubs');
    playercardaddValue=10;
    
elseif playercardadd==51
    playercardaddName=('King of Diamonds');
    playercardaddValue=10;
    
elseif playercardadd==52
    playercardaddName=('King of Hearts');
    playercardaddValue=10;
end    

%Class for the decision of the Player from "dealblackjack"
clc;
%while player is hitting, hittingCheck variable is established, while loop
%is created for player count value.
hittingCheck=0;
%makes sure player hasn't busted or gotten blackjack, makes sure
%hittingCheck is also still zero. 
while playercount < 21 && hittingCheck==0
    %Displays a decision screen and an input option
fprintf('The Dealer shows: %s\n\n',dealerCard1Name)
fprintf('Your total is %d\n\n',playercount)
fprintf('What is your decision?\n')
    fprintf('1: Hit\n')
    fprintf('2: Stay\n')
    %takes input and executes by storing it in variable "playerChoice"
    playerChoice=input('Player Selects: ','s');
        %if statement created for input decision
        
        %decision for if player hits
        if  playerChoice=='1' 
            %Adds 4 for the creation of new matrix location
            playCard1=(playCard1+4);
            %establishes value to be added to the playercount
            playercardadd=(dealCard(playCard1));
            %Goes to class created for the player hitting process, returns
            %with name and value of card
            PlayerHitting
            %Clears
            clc;
            %Displays new card name and value, adds it to total
            fprintf('The Dealer hit you with %s\n\n',playercardaddName)
            playercount=(playercount+playercardaddValue);
            fprintf('You now have a total of %d\n',playercount)
            %Press any key to continue using pause, returns back to top of
            %class
            fprintf('\nPress any key to continue.\n')
            pause;
            PlayerDecisions
        
            %Decision process for if player decides to stay
        elseif playerChoice=='2'
               %displays screen for if player stays
               clc;
               fprintf('You stood with %d\n',playercount)
               %any key to continue using pause
               fprintf('\nPress any key to continue.\n')
               pause;
               %exits while loop and returns to "dealblackjack" from where
               %it was called
               
               %hittingCheck is set to 1 so the while loop is exited.
               hittingCheck=1;
        else 
               %handles the error if the value inputted was not 1 or 2.
               %Displays error message to the user
               clc;
               fprintf('Dealer says: "That decision is against the rules."\n')
               fprintf('        "You can either choose to hit or stay."\n')
               %Press any key using pause; calls PlayerDecisions to return
               %back to top of class
               fprintf('/nPress any key to try again.\n')
               pause;
               PlayerDecisions
        end
end

%checks to see if the player has blackjack or bust

%First if statement checks for blackjack
if playercount==21
    %the player has blackjack
    %Clears
    clc;
    %Displays blackjack winning screen for player
    fprintf('Dealer: "you have hit a Blackjack!"\n')
    fprintf('        "I will now give you what you have won."\n\n')
    %Assigns value of winnings to variable, adds that to players total
    winBet=(playerBet*2);
    moneyvalue=(moneyvalue+winBet);
    %Display for winnings
    fprintf('The dealer gives you %0.2f dollars.\n',winBet)
    fprintf('You now have %0.2f dollars.\n\n',moneyvalue)
    fprintf('Press any key to return to continue.')
    %press any key using pause; returns back to beginning of game using
    %playblackjack
    pause;
    playblackjack
    
    %Scenario for if the player has busted
elseif playercount > 21
    %the player has busted
    %Clears
    clc;
    %Displays screen for player losing from bust.
    fprintf('Dealer says: "You busted."\n')
    fprintf('You lost your bet of %0.2f dollars.\n',playerBet)
    %bet value is not returned to player
    fprintf('You now have %0.2f dollars.\n',moneyvalue)
    fprintf('Press any key to continue.\n')
    %Press any key function using pause; returns back to beginning of game
    %using playblackjack
    pause;
    playblackjack
    
else
    %player stayed with less than 21, continues to dealblackjack from where
    %class was called to continue evaluation
    %Clears
    clc;
end
%returns to dealblackjack

%this is the blackjack game itself

clc;
%Displayed menu options, along with input option
fprintf('What would you like to do?\n\n')
fprintf('1:Place your bet\n')
fprintf('2:"How to play Blackjack"\n')
fprintf('3:Choose another game\n')
playDecision=input('','s');

%if statements dependent on user input
%First decision is for playing the game
if playDecision== '1' 
    clc;
fprintf('Dealer says: "How much are you betting?"\n\n')

%shows how much money player has
fprintf('You have %0.2f dollars.\n\n',moneyvalue)

%Shows restrictions on betting amount, creates input for user.

fprintf('Please bet an amount with no more than 2 decimal places.\n')
playerBet=input('Your bet: ','s');

%checks that the user input betting amount is valid
%Creation of arrays to check value for validity
decimal='.';
numArray=['1' '2' '3' '4' '5' '6' '7' '8' '9' '0'];
nonzeroArray=['1' '2' '3' '4' '5' '6' '7' '8' '9'];
%first checks for a decimal using ismember() function
decimal_1=ismember(playerBet,decimal);
checkDecimal_1=sum(decimal_1);
%can only have 1 decimal so checkdec1 must equal 1
if checkDecimal_1==1
    %has only 1 decimal, is valid, check for numbers
    %now it checks that the players bet > 0
    minimumBet=ismember(playerBet,nonzeroArray);
    checkminimumbet=sum(minimumBet);
    if checkminimumbet==0
        %in this case there are no numbers other than zero,reports an error
        %in the input.
        clc;
        fprintf('Dealer says: "You must bet at least a penny to play Blackjack."\n\n')
        fprintf('Please bet a non-zero number, the minimum is 0.01 \n')
        fprintf('Press any key to try again.\n')
        pause;
        clc;
        playblackjack
    else 
        %in this case it is known the bet is non-zero, now it checks for a
        %valid number using ismember() again.
        betLength=length(playerBet);
        numberBet=ismember(playerBet,numArray);
        checknumberbet=sum(numberBet);
        %Checks to see that the number is valid by comparing the amount of
        %approvals from numArray values to the length of the input.
        if checknumberbet==(betLength-1)
            %the number is a valid number, convert to numeric using str2double() for calculations
            playerBet=str2double(playerBet)
        else
            %not a valid number in this case, sends message statement to
            %user.
            clc;
            fprintf('Dealer says: "You can''t bet that!\n\n')
            fprintf('You must enter a real number, with a maximum of 2 decimal places.\n')
            fprintf('Press any key to try again.\n')
            pause;
            clc;
            playblackjack
        end
    end
    %same thing, but checks for the input when there is no decimal
    %inputted, which is also acceptable. (Not efficient code but
    %effective). 
elseif checkDecimal_1==0
    %no decimal, is possible, check for bet >0
   minimumBet=ismember(playerBet,nonzeroArray);
    checkminimumbet=sum(minimumBet);
    if checkminimumbet==0
        %no numbers other than zero, not a noon-zero number, reports error
        %message
        clc;
        fprintf('Dealer says: "You must bet at least a penny to play Black Jack."\n\n')
        fprintf('Please bet a non-zero number,the minimum is 0.01\n')
        fprintf('Press any key to try again.\n')
        pause;
        clc;
        playblackjack
    else 
        %is a non-zero bet, now checks for a valid number value
        betLength=length(playerBet);
        numberBet=ismember(playerBet,numArray);
        checknumberbet=sum(numberBet);
        if checknumberbet==(betLength)
            %is valid number, convert to numeric using str2double() for calculations
            playerBet=str2double(playerBet)
        else
            %not valid input, reports error message
            clc;
            fprintf('Dealer says: "You can''t bet that!\n\n')
            fprintf('You must enter a real number, with a maximum of 2 decimal places.\n')
            fprintf('Press any key to try again.\n')
            pause;
            clc;
            playblackjack
        end
    end
    
else 
    %has more than 1 decimal, invalid, sends error message to user.
    clc;
    fprintf('Dealer says: "You can''t bet that!\n\n')
    fprintf('You must enter a real number, with a maximum of 2 decimal places.\n')
    fprintf('Press any key to try again.\n')
   
    pause;
    clc;
    playblackjack
end

%takes player input as number, then determines if the player has enough
%money to play in the round
checkBet=((moneyvalue)-(playerBet));
if checkBet >= 0 
    %has enough money, starts the game
    clc;
    %Takes bet value away from total
    moneyvalue=(moneyvalue-playerBet);
    fprintf('You bet %0.2f dollars\n\n',playerBet)
    fprintf('You now have %0.2f dollars remaining\n\n',moneyvalue)
    fprintf('Press any key to continue\n')
    pause;
    %laying out the cards in random order using randperm() function
dealCard=(randperm(52,52));

%setting the starting variables for game functionality.

%count total for both the dealer and player is established.
dealercount=(0);
playercount=(0);

% the matrix locators  are created for each card of the player and dealer 
% (Necessary for the concept of "turns" for each card turn.
dealCard1=(1);
dealCard2=(2);
playCard1=(3);
playCard2=(4);

%assigns cards using the dealCard function established by randperm(). By
%splitting matrices, it prevents card counting as the deck is new for each
%new "card" in each matrix
dealercard1=dealCard(dealCard1);
dealercard2=dealCard(dealCard2);
playercard1=dealCard(playCard1);
playercard2=dealCard(playCard2);
%Class created for the dealing operations of the game. Does dealing after
%first two cards are dealt
    dealblackjack
else
    %does not have enough money, reports an error message to the user.
    clc;
    fprintf('Dealer says: "Betting on credit is not allowed. Place a bet that you can afford please."\n\n')
    fprintf('Press any key to continue.\n')
    pause;
    clc;
    playblackjack
    
end
%Original if statement for menu options, if not choosing to play the game,
%inputting two will display the rules of the game.
elseif playDecision== '2'
    %displays rules
    clc;
    fprintf('Dealer says: "You want to get the total value of your cards to 21. I am\n')
    fprintf('        trying to do the same. Before I deal the cards, you have to place\n')
    fprintf('        a bet. After your bet if placed, I deal two cards to you and two\n')
    fprintf('        to myself. I cannot see any of your cards, but you can see one of\n')
    fprintf('        mine. You can either hit or stay, hitting would be drawing another\n')
    fprintf('        card, staying would be remaining at the value you already have. As\n')
    fprintf('        long as you are under 21, you can continue to hit in hopes of reaching\n')
    fprintf('        21. If you reach 21, you win, and I pay you. If you go over 21, however,\n')
    fprintf('        you bust, which means you lose. If you decide to stay on a value lower\n')
    fprintf('        than 21, I will begin to hit in hopes of getting 21 or a value closer\n')
    fprintf('        to it. In Black Jack, the dealer is required to hit on any value 16 or\n')
    fprintf('        lower, and is required to stay on any value 17 or higher. If the dealer\n')
    fprintf('        busts or has a lower value than you, you win. If we are tied, however,\n')
    fprintf('        I win. The value of each card is just the number on them, except for\n')
    fprintf('        face cards. Each face card is worth 10, except for Aces. Aces are worth\n')
    fprintf('        11, or 1 if 11 would cause you to bust. Have fun playing!!"\n\n')
    fprintf('Press any key to go back.\n')
    %Pausing allows for the concept of "Press any key to continue"
    %throughout the gameplay
    pause;
    %Restarts back at the top of this class
    playblackjack
    
    %Final option of if else statement
elseif playDecision== '3'
    %returns to "menu" option to play other games (Hangman, Over-Under
    %Seven)
    clc;
    fprintf('Dealer says: "Come back again!"\n')
    fprintf('Press any key to continue')
    pause;
    clc;
    return
    
else 
    %handles the error input from user.
    clc;
    fprintf('That is not an option.\n')
    fprintf('Press any key to try again\n')
    pause;
    %resets at top of class.
    playblackjack
end

%The "if loop" for the BlackJack game. Decides what the value and name of each card
%is dependent on the number returned and stored by randperm().

fprintf('The Dealer shuffles the cards and deals them out.\n\n')
%checks number for dealercard1 in large, inefficient if else statement that covers all values to 52. The
%same process is done in this class for the next dealercard and the two
%playercards. Many lines of code, but nonetheless functions effectively. 
if dealercard1==1
    dealerCard1Name=('Ace of Spades');
    %determines if count for the Ace should use 11 or 1
    checkDealerCard=((dealercount)+11);
    %Checks of the dealer will bust
    if checkDealerCard > 21 
        dealerCard1Value=1;
    else 
        dealerCard1Value=11;
    end
    
elseif dealercard1==2
    dealerCard1Name=('Ace of Clubs');
    %determines if count for the Ace should use 11 or 1
    checkDealerCard=((dealercount)+11);
    %Checks if the dealer will bust
    if checkDealerCard > 21 
        dealerCard1Value=1;
    else 
        dealerCard1Value=11;
    end
    
elseif dealercard1==3
    dealerCard1Name=('Ace of Diamonds');
    %determines if count for the Ace should use 11 or 1
    checkDealerCard=((dealercount)+11);
    %Checks if the dealer will bust
    if checkDealerCard > 21 
        dealerCard1Value=1;
    else 
        dealerCard1Value=11;
    end
    
elseif dealercard1==4
    dealerCard1Name=('Ace of Hearts');
    %determines if count for the Ace should use 11 or 1
    checkDealerCard=((dealercount)+11);
    %Checks if the dealer will bust
    if checkDealerCard > 21 
        dealerCard1Value=1;
    else 
        dealerCard1Value=11;
    end
    
        %Rest of values are matched with a specific card of the 52-card deck
elseif dealercard1==5
    dealerCard1Name=('Two of Spades');
    dealerCard1Value=2;
    
elseif dealercard1==6
    dealerCard1Name=('Two of Clubs');
    dealerCard1Value=2;
    
elseif dealercard1==7
    dealerCard1Name=('Two of Diamonds');
    dealerCard1Value=2;
    
elseif dealercard1==8
    dealerCard1Name=('Two of Hearts');
    dealerCard1Value=2;
    
elseif dealercard1==9
    dealerCard1Name=('Three of Spades');
    dealerCard1Value=3;
    
elseif dealercard1==10
    dealerCard1Name=('Three of Clubs');
    dealerCard1Value=3;
    
elseif dealercard1==11
    dealerCard1Name=('Three of Diamonds');
    dealerCard1Value=3;
    
elseif dealercard1==12
    dealerCard1Name=('Three of Hearts');
    dealerCard1Value=3;
    
elseif dealercard1==13
    dealerCard1Name=('Four of Spades');
    dealerCard1Value=4;
    
elseif dealercard1==14
    dealerCard1Name=('Four of Clubs');
    dealerCard1Value=4;
    
elseif dealercard1==15
    dealerCard1Name=('Four of Diamonds');
    dealerCard1Value=4;
    
elseif dealercard1==16
    dealerCard1Name=('Four of Hearts');
    dealerCard1Value=4;
    
elseif dealercard1==17
    dealerCard1Name=('Five of Spades');
    dealerCard1Value=5;
    
elseif dealercard1==18
    dealerCard1Name=('Five of Clubs');
    dealerCard1Value=5;
    
elseif dealercard1==19
    dealerCard1Name=('Five of Diamonds');
    dealerCard1Value=5;
    
elseif dealercard1==20
    dealerCard1Name=('Five of Hearts');
    dealerCard1Value=5;
    
elseif dealercard1==21
    dealerCard1Name=('Six of Spades');
    dealerCard1Value=6;
    
elseif dealercard1==22
    dealerCard1Name=('Six of Clubs');
    dealerCard1Value=6;
    
elseif dealercard1==23
    dealerCard1Name=('Six of Diamonds');
    dealerCard1Value=6;
    
elseif dealercard1==24
    dealerCard1Name=('Six of Hearts');
    dealerCard1Value=6;
    
elseif dealercard1==25
    dealerCard1Name=('Seven of Spades');
    dealerCard1Value=7;
    
elseif dealercard1==26
    dealerCard1Name=('Seven of Clubs');
    dealerCard1Value=7;
    
elseif dealercard1==27
    dealerCard1Name=('Seven of Diamonds');
    dealerCard1Value=7;
    
elseif dealercard1==28
    dealerCard1Name=('Seven of Hearts');
    dealerCard1Value=7;
    
elseif dealercard1==29
    dealerCard1Name=('Eight of Spades');
    dealerCard1Value=8;
    
elseif dealercard1==30
    dealerCard1Name=('Eight of Clubs');
    dealerCard1Value=8;
    
elseif dealercard1==31
    dealerCard1Name=('Eight of Diamonds');
    dealerCard1Value=8;
    
elseif dealercard1==32
    dealerCard1Name=('Eight of Hearts');
    dealerCard1Value=8;
    
elseif dealercard1==33
    dealerCard1Name=('Nine of Spades');
    dealerCard1Value=9;
    
elseif dealercard1==34
    dealerCard1Name=('Nine of Clubs');
    dealerCard1Value=9;
    
elseif dealercard1==35
    dealerCard1Name=('Nine of Diamonds');
    dealerCard1Value=9;
    
elseif dealercard1==36
    dealerCard1Name=('Nine of Hearts');
    dealerCard1Value=9;
    
elseif dealercard1==37
    dealerCard1Name=('Ten of Spades');
    dealerCard1Value=10;
    
elseif dealercard1==38
    dealerCard1Name=('Ten of Clubs');
    dealerCard1Value=10;
    
elseif dealercard1==39
    dealerCard1Name=('Ten of Diamonds');
    dealerCard1Value=10;
    
elseif dealercard1==40
    dealerCard1Name=('Ten of Hearts');
    dealerCard1Value=10;
    
elseif dealercard1==41
    dealerCard1Name=('Jack of Spades');
    dealerCard1Value=10;
    
elseif dealercard1==42
    dealerCard1Name=('Jack of Clubs');
    dealerCard1Value=10;
    
elseif dealercard1==43
    dealerCard1Name=('Jack of Diamonds');
    dealerCard1Value=10;
    
elseif dealercard1==44
    dealerCard1Name=('Jack of Hearts');
    dealerCard1Value=10;
    
elseif dealercard1==45
    dealerCard1Name=('Queen of Spades');
    dealerCard1Value=10;
    
elseif dealercard1==46
    dealerCard1Name=('Queen of Clubs');
    dealerCard1Value=10;
    
elseif dealercard1==47
    dealerCard1Name=('Queen of Diamonds');
    dealerCard1Value=10;
    
elseif dealercard1==48
    dealerCard1Name=('Queen of Hearts');
    dealerCard1Value=10;
    
elseif dealercard1==49
    dealerCard1Name=('King of Spades');
    dealerCard1Value=10;
    
elseif dealercard1==50
    dealerCard1Name=('King of Clubs');
    dealerCard1Value=10;
    
elseif dealercard1==51
    dealerCard1Name=('King of Diamonds');
    dealerCard1Value=10;
    
elseif dealercard1==52
    dealerCard1Name=('King of Hearts');
    dealerCard1Value=10;
end

%if statement for the dealercard2
if dealercard2==1
    dealerCard2Name=('Ace of Spades');
    %determines if count for the Ace should use 11 or 1
    checkDealerCard=((dealercount)+11);
    %Checks if the dealer will bust
    if checkDealerCard > 21 
        dealerCard2Value=1;
    else 
        dealerCard2Value=11;
    end
    
elseif dealercard2==2
    dealerCard2Name=('Ace of Clubs');
    %determines if count for the Ace should use 11 or 1
    checkDealerCard=((dealercount)+11);
    %Checks if the dealer will bust
    if checkDealerCard > 21 
        dealerCard2Value=1;
    else 
        dealerCard2Value=11;
    end
    
elseif dealercard2==3
    dealerCard2Name=('Ace of Diamonds');
    %determines if count for the Ace should use 11 or 1
    checkDealerCard=((dealercount)+11);
    %Checks if the dealer will bust
    if checkDealerCard > 21 
        dealerCard2Value=1;
    else 
        dealerCard2Value=11;
    end
    
elseif dealercard2==4
    dealerCard2Name=('Ace of Hearts');
    %determines if count for the Ace should use 11 or 1
    checkDealerCard=((dealercount)+11);
    %Checks if the dealer will bust
    if checkDealerCard > 21 
        dealerCard2Value=1;
    else 
        dealerCard2Value=11;
    end
   
        %Rest of values are matched with a specific card of the 52-card deck
elseif dealercard2==5
    dealerCard2Name=('Two of Spades');
    dealerCard2Value=2;
    
elseif dealercard2==6
    dealerCard2Name=('Two of Clubs');
    dealerCard2Value=2;
    
elseif dealercard2==7
    dealerCard2Name=('Two of Diamonds');
    dealerCard2Value=2;
    
elseif dealercard2==8
    dealerCard2Name=('Two of Hearts');
    dealerCard2Value=2;
    
elseif dealercard2==9
    dealerCard2Name=('Three of Spades');
    dealerCard2Value=3;
    
elseif dealercard2==10
    dealerCard2Name=('Three of Clubs');
    dealerCard2Value=3;
    
elseif dealercard2==11
    dealerCard2Name=('Three of Diamonds');
    dealerCard2Value=3;
    
elseif dealercard2==12
    dealerCard2Name=('Three of Hearts');
    dealerCard2Value=3;
    
elseif dealercard2==13
    dealerCard2Name=('Four of Spades');
    dealerCard2Value=4;
    
elseif dealercard2==14
    dealerCard2Name=('Four of Clubs');
    dealerCard2Value=4;
    
elseif dealercard2==15
    dealerCard2Name=('Four of Diamonds');
    dealerCard2Value=4;
    
elseif dealercard2==16
    dealerCard2Name=('Four of Hearts');
    dealerCard2Value=4;
    
elseif dealercard2==17
    dealerCard2Name=('Five of Spades');
    dealerCard2Value=5;
    
elseif dealercard2==18
    dealerCard2Name=('Five of Clubs');
    dealerCard2Value=5;
    
elseif dealercard2==19
    dealerCard2Name=('Five of Diamonds');
    dealerCard2Value=5;
    
elseif dealercard2==20
    dealerCard2Name=('Five of Hearts');
    dealerCard2Value=5;
    
elseif dealercard2==21
    dealerCard2Name=('Six of Spades');
    dealerCard2Value=6;
    
elseif dealercard2==22
    dealerCard2Name=('Six of Clubs');
    dealerCard2Value=6;
    
elseif dealercard2==23
    dealerCard2Name=('Six of Diamonds');
    dealerCard2Value=6;
    
elseif dealercard2==24
    dealerCard2Name=('Six of Hearts');
    dealerCard2Value=6;
    
elseif dealercard2==25
    dealerCard2Name=('Seven of Spades');
    dealerCard2Value=7;
    
elseif dealercard2==26
    dealerCard2Name=('Seven of Clubs');
    dealerCard2Value=7;
    
elseif dealercard2==27
    dealerCard2Name=('Seven of Diamonds');
    dealerCard2Value=7;
    
elseif dealercard2==28
    dealerCard2Name=('Seven of Hearts');
    dealerCard2Value=7;
    
elseif dealercard2==29
    dealerCard2Name=('Eight of Spades');
    dealerCard2Value=8;
    
elseif dealercard2==30
    dealerCard2Name=('Eight of Clubs');
    dealerCard2Value=8;
    
elseif dealercard2==31
    dealerCard2Name=('Eight of Diamonds');
    dealerCard2Value=8;
    
elseif dealercard2==32
    dealerCard2Name=('Eight of Hearts');
    dealerCard2Value=8;
    
elseif dealercard2==33
    dealerCard2Name=('Nine of Spades');
    dealerCard2Value=9;
    
elseif dealercard2==34
    dealerCard2Name=('Nine of Clubs');
    dealerCard2Value=9;
    
elseif dealercard2==35
    dealerCard2Name=('Nine of Diamonds');
    dealerCard2Value=9;
    
elseif dealercard2==36
    dealerCard2Name=('Nine of Hearts');
    dealerCard2Value=9;
    
elseif dealercard2==37
    dealerCard2Name=('Ten of Spades');
    dealerCard2Value=10;
    
elseif dealercard2==38
    dealerCard2Name=('Ten of Clubs');
    dealerCard2Value=10;
    
elseif dealercard2==39
    dealerCard2Name=('Ten of Diamonds');
    dealerCard2Value=10;
    
elseif dealercard2==40
    dealerCard2Name=('Ten of Hearts');
    dealerCard2Value=10;
    
elseif dealercard2==41
    dealerCard2Name=('Jack of Spades');
    dealerCard2Value=10;
    
elseif dealercard2==42
    dealerCard2Name=('Jack of Clubs');
    dealerCard2Value=10;
    
elseif dealercard2==43
    dealerCard2Name=('Jack of Diamonds');
    dealerCard2Value=10;
    
elseif dealercard2==44
    dealerCard2Name=('Jack of Hearts');
    dealerCard2Value=10;
    
elseif dealercard2==45
    dealerCard2Name=('Queen of Spades');
    dealerCard2Value=10;
    
elseif dealercard2==46
    dealerCard2Name=('Queen of Clubs');
    dealerCard2Value=10;
    
elseif dealercard2==47
    dealerCard2Name=('Queen of Diamonds');
    dealerCard2Value=10;
    
elseif dealercard2==48
    dealerCard2Name=('Queen of Hearts');
    dealerCard2Value=10;
    
elseif dealercard2==49
    dealerCard2Name=('King of Spades');
    dealerCard2Value=10;
    
elseif dealercard2==50
    dealerCard2Name=('King of Clubs');
    dealerCard2Value=10;
    
elseif dealercard2==51
    dealerCard2Name=('King of Diamonds');
    dealerCard2Value=10;
    
elseif dealercard2==52
    dealerCard2Name=('King of Hearts');
    dealerCard2Value=10;
end

%if statement for the playercard1 
if playercard1==1
    playerCard1Name=('Ace of Spades');
    %determines if count for the Ace should use 11 or 1
    checkPlayerCard=((playercount)+11);
    %Checks if player will bust
    if checkPlayerCard > 21 
        playerCard1Value=1;
    else 
        playerCard1Value=11;
    end
    
elseif playercard1==2
    playerCard1Name=('Ace of Clubs');
    %determines if count for the Ace should use 11 or 1
    checkPlayerCard=((playercount)+11);
    %Checks if player will bust
    if checkPlayerCard > 21 
        playerCard1Value=1;
    else 
        playerCard1Value=11;
    end
    
elseif playercard1==3
    playerCard1Name=('Ace of Diamonds');
    %determines if count for the Ace should use 11 or 1
    checkPlayerCard=((playercount)+11);
    %Checks if player will bust
    if checkPlayerCard > 21 
        playerCard1Value=1;
    else 
        playerCard1Value=11;
    end
    
elseif playercard1==4
    playerCard1Name=('Ace of Hearts');
    %determines if count for the Ace should use 11 or 1
    checkPlayerCard=((playercount)+11);
    %Checks if player will bust
    if checkPlayerCard > 21 
        playerCard1Value=1;
    else 
        playerCard1Value=11;
    end
    
    %Rest of values are matched with a specific card of the 52-card deck
elseif playercard1==5
    playerCard1Name=('Two of Spades');
    playerCard1Value=2;
    
elseif playercard1==6
    playerCard1Name=('Two of Clubs');
    playerCard1Value=2;
    
elseif playercard1==7
    playerCard1Name=('Two of Diamonds');
    playerCard1Value=2;
    
elseif playercard1==8
    playerCard1Name=('Two of Hearts');
    playerCard1Value=2;
    
elseif playercard1==9
    playerCard1Name=('Three of Spades');
    playerCard1Value=3;
    
elseif playercard1==10
    playerCard1Name=('Three of Clubs');
    playerCard1Value=3;
    
elseif playercard1==11
    playerCard1Name=('Three of Diamonds');
    playerCard1Value=3;
    
elseif playercard1==12
    playerCard1Name=('Three of Hearts');
    playerCard1Value=3;
    
elseif playercard1==13
    playerCard1Name=('Four of Spades');
    playerCard1Value=4;
    
elseif playercard1==14
    playerCard1Name=('Four of Clubs');
    playerCard1Value=4;
    
elseif playercard1==15
    playerCard1Name=('Four of Diamonds');
    playerCard1Value=4;
    
elseif playercard1==16
    playerCard1Name=('Four of Hearts');
    playerCard1Value=4;
    
elseif playercard1==17
    playerCard1Name=('Five of Spades');
    playerCard1Value=5;
    
elseif playercard1==18
    playerCard1Name=('Five of Clubs');
    playerCard1Value=5;
    
elseif playercard1==19
    playerCard1Name=('Five of Diamonds');
    playerCard1Value=5;
    
elseif playercard1==20
    playerCard1Name=('Five of Hearts');
    playerCard1Value=5;
    
elseif playercard1==21
    playerCard1Name=('Six of Spades');
    playerCard1Value=6;
    
elseif playercard1==22
    playerCard1Name=('Six of Clubs');
    playerCard1Value=6;
    
elseif playercard1==23
    playerCard1Name=('Six of Diamonds');
    playerCard1Value=6;
    
elseif playercard1==24
    playerCard1Name=('Six of Hearts');
    playerCard1Value=6;
    
elseif playercard1==25
    playerCard1Name=('Seven of Spades');
    playerCard1Value=7;
    
elseif playercard1==26
    playerCard1Name=('Seven of Clubs');
    playerCard1Value=7;
    
elseif playercard1==27
    playerCard1Name=('Seven of Diamonds');
    playerCard1Value=7;
    
elseif playercard1==28
    playerCard1Name=('Seven of Hearts');
    playerCard1Value=7;
    
elseif playercard1==29
    playerCard1Name=('Eight of Spades');
    playerCard1Value=8;
    
elseif playercard1==30
    playerCard1Name=('Eight of Clubs');
    playerCard1Value=8;
    
elseif playercard1==31
    playerCard1Name=('Eight of Diamonds');
    playerCard1Value=8;
    
elseif playercard1==32
    playerCard1Name=('Eight of Hearts');
    playerCard1Value=8;
    
elseif playercard1==33
    playerCard1Name=('Nine of Spades');
    playerCard1Value=9;
    
elseif playercard1==34
    playerCard1Name=('Nine of Clubs');
    playerCard1Value=9;
    
elseif playercard1==35
    playerCard1Name=('Nine of Diamonds');
    playerCard1Value=9;
    
elseif playercard1==36
    playerCard1Name=('Nine of Hearts');
    playerCard1Value=9;
    
elseif playercard1==37
    playerCard1Name=('Ten of Spades');
    playerCard1Value=10;
    
elseif playercard1==38
    playerCard1Name=('Ten of Clubs');
    playerCard1Value=10;
    
elseif playercard1==39
    playerCard1Name=('Ten of Diamonds');
    playerCard1Value=10;
    
elseif playercard1==40
    playerCard1Name=('Ten of Hearts');
    playerCard1Value=10;
    
elseif playercard1==41
    playerCard1Name=('Jack of Spades');
    playerCard1Value=10;
    
elseif playercard1==42
    playerCard1Name=('Jack of Clubs');
    playerCard1Value=10;
    
elseif playercard1==43
    playerCard1Name=('Jack of Diamonds');
    playerCard1Value=10;
    
elseif playercard1==44
    playerCard1Name=('Jack of Hearts');
    playerCard1Value=10;
    
elseif playercard1==45
    playerCard1Name=('Queen of Spades');
    playerCard1Value=10;
    
elseif playercard1==46
    playerCard1Name=('Queen of Clubs');
    playerCard1Value=10;
    
elseif playercard1==47
    playerCard1Name=('Queen of Diamonds');
    playerCard1Value=10;
    
elseif playercard1==48
    playerCard1Name=('Queen of Hearts');
    playerCard1Value=10;
    
elseif playercard1==49
    playerCard1Name=('King of Spades');
    playerCard1Value=10;
    
elseif playercard1==50
    playerCard1Name=('King of Clubs');
    playerCard1Value=10;
    
elseif playercard1==51
    playerCard1Name=('King of Diamonds');
    playerCard1Value=10;
    
elseif playercard1==52
    playerCard1Name=('King of Hearts');
    playerCard1Value=10;
end

%if statement for the playercard2
if playercard2==1
    playerCard2Name=('Ace of Spades');
    %determines if count for the Ace should use 11 or 1
    checkPlayerCard=((playercount)+11);
    %Checks if player will bust
    if checkPlayerCard > 21 
        playerCard2Value=1;
    else 
        playerCard2Value=11;
    end
    
elseif playercard2==2
    playerCard2Name=('Ace of Clubs');
    %determines if count for the Ace should use 11 or 1
    checkPlayerCard=((playercount)+11);
    %Checks if player will bust
    if checkPlayerCard > 21 
        playerCard2Value=1;
    else 
        playerCard2Value=11;
    end
    
elseif playercard2==3
    playerCard2Name=('Ace of Diamonds');
    %determines if count for the Ace should use 11 or 1
    checkPlayerCard=((playercount)+11);
    %Checks if player will bust
    if checkPlayerCard > 21 
        playerCard2Value=1;
    else 
        playerCard2Value=11;
    end
    
elseif playercard2==4
    playerCard2Name=('Ace of Hearts');
    %determines if count for the Ace should use 11 or 1
    checkPlayerCard=((playercount)+11);
    %Checks if player will bust
    if checkPlayerCard > 21 
        playerCard2Value=1;
    else 
        playerCard2Value=11;
    end
    
        %Rest of values are matched with a specific card of the 52-card deck
elseif playercard2==5
    playerCard2Name=('Two of Spades');
    playerCard2Value=2;
    
elseif playercard2==6
    playerCard2Name=('Two of Clubs');
    playerCard2Value=2;
    
elseif playercard2==7
    playerCard2Name=('Two of Diamonds');
    playerCard2Value=2;
    
elseif playercard2==8
    playerCard2Name=('Two of Hearts');
    playerCard2Value=2;
    
elseif playercard2==9
    playerCard2Name=('Three of Spades');
    playerCard2Value=3;
    
elseif playercard2==10
    playerCard2Name=('Three of Clubs');
    playerCard2Value=3;
    
elseif playercard2==11
    playerCard2Name=('Three of Diamonds');
    playerCard2Value=3;
    
elseif playercard2==12
    playerCard2Name=('Three of Hearts');
    playerCard2Value=3;
    
elseif playercard2==13
    playerCard2Name=('Four of Spades');
    playerCard2Value=4;
    
elseif playercard2==14
    playerCard2Name=('Four of Clubs');
    playerCard2Value=4;
    
elseif playercard2==15
    playerCard2Name=('Four of Diamonds');
    playerCard2Value=4;
    
elseif playercard2==16
    playerCard2Name=('Four of Hearts');
    playerCard2Value=4;
    
elseif playercard2==17
    playerCard2Name=('Five of Spades');
    playerCard2Value=5;
    
elseif playercard2==18
    playerCard2Name=('Five of Clubs');
    playerCard2Value=5;
    
elseif playercard2==19
    playerCard2Name=('Five of Diamonds');
    playerCard2Value=5;
    
elseif playercard2==20
    playerCard2Name=('Five of Hearts');
    playerCard2Value=5;
    
elseif playercard2==21
    playerCard2Name=('Six of Spades');
    playerCard2Value=6;
    
elseif playercard2==22
    playerCard2Name=('Six of Clubs');
    playerCard2Value=6;
    
elseif playercard2==23
    playerCard2Name=('Six of Diamonds');
    playerCard2Value=6;
    
elseif playercard2==24
    playerCard2Name=('Six of Hearts');
    playerCard2Value=6;
    
elseif playercard2==25
    playerCard2Name=('Seven of Spades');
    playerCard2Value=7;
    
elseif playercard2==26
    playerCard2Name=('Seven of Clubs');
    playerCard2Value=7;
    
elseif playercard2==27
    playerCard2Name=('Seven of Diamonds');
    playerCard2Value=7;
    
elseif playercard2==28
    playerCard2Name=('Seven of Hearts');
    playerCard2Value=7;
    
elseif playercard2==29
    playerCard2Name=('Eight of Spades');
    playerCard2Value=8;
    
elseif playercard2==30
    playerCard2Name=('Eight of Clubs');
    playerCard2Value=8;
    
elseif playercard2==31
    playerCard2Name=('Eight of Diamonds');
    playerCard2Value=8;
    
elseif playercard2==32
    playerCard2Name=('Eight of Hearts');
    playerCard2Value=8;
    
elseif playercard2==33
    playerCard2Name=('Nine of Spades');
    playerCard2Value=9;
    
elseif playercard2==34
    playerCard2Name=('Nine of Clubs');
    playerCard2Value=9;
    
elseif playercard2==35
    playerCard2Name=('Nine of Diamonds');
    playerCard2Value=9;
    
elseif playercard2==36
    playerCard2Name=('Nine of Hearts');
    playerCard2Value=9;
    
elseif playercard2==37
    playerCard2Name=('Ten of Spades');
    playerCard2Value=10;
    
elseif playercard2==38
    playerCard2Name=('Ten of Clubs');
    playerCard2Value=10;
    
elseif playercard2==39
    playerCard2Name=('Ten of Diamonds');
    playerCard2Value=10;
    
elseif playercard2==40
    playerCard2Name=('Ten of Hearts');
    playerCard2Value=10;
    
elseif playercard2==41
    playerCard2Name=('Jack of Spades');
    playerCard2Value=10;
    
elseif playercard2==42
    playerCard2Name=('Jack of Clubs');
    playerCard2Value=10;
    
elseif playercard2==43
    playerCard2Name=('Jack of Diamonds');
    playerCard2Value=10;
    
elseif playercard2==44
    playerCard2Name=('Jack of Hearts');
    playerCard2Value=10;
    
elseif playercard2==45
    playerCard2Name=('Queen of Spades');
    playerCard2Value=10;
    
elseif playercard2==46
    playerCard2Name=('Queen of Clubs');
    playerCard2Value=10;
    
elseif playercard2==47
    playerCard2Name=('Queen of Diamonds');
    playerCard2Value=10;
    
elseif playercard2==48
    playerCard2Name=('Queen of Hearts');
    playerCard2Value=10;
    
elseif playercard2==49
    playerCard2Name=('King of Spades');
    playerCard2Value=10;
    
elseif playercard2==50
    playerCard2Name=('King of Clubs');
    playerCard2Value=10;
    
elseif playercard2==51
    playerCard2Name=('King of Diamonds');
    playerCard2Value=10;
    
elseif playercard2==52
    playerCard2Name=('King of Hearts');
    playerCard2Value=10;
end

%cards assigned, calculates total counts and assigns the value to the
%counts of dealer and player
dealercount=((dealerCard1Value)+(dealerCard2Value));
playercount=((playerCard1Value)+(playerCard2Value));

%end of assignments and calculations, returns to "dealblackjack"

moneyTotalChar = num2str(moneyTotal);
                    if length(moneyTotalChar) == 4
                        foreground(xCoinCount, yCoinCountHundreds) = displayGoldNine;
                        foreground(xCoinCount, yCoinCountTens) = displayGoldNine;
                        foreground(xCoinCount, yCoinCountOnes) = displayGoldNine;
                        foregroundShop(xCoinCount, yCoinCountHundredsShop) = displayGoldNineShop;
                        foregroundShop(xCoinCount, yCoinCountTensShop) = displayGoldNineShop;
                        foregroundShop(xCoinCount, yCoinCountOnesShop) = displayGoldNineShop;
                    elseif length(moneyTotalChar) == 3
                        if moneyTotalChar(1) == '1'
                            foreground(xCoinCount, yCoinCountHundreds) = displayOne;
                            foregroundShop(xCoinCount, yCoinCountHundredsShop) = displayOneShop;
                        elseif moneyTotalChar(1) == '2'
                            foreground(xCoinCount, yCoinCountHundreds) = displayTwo;
                            foregroundShop(xCoinCount, yCoinCountHundredsShop) = displayTwoShop;
                        elseif moneyTotalChar(1) == '3'
                            foreground(xCoinCount, yCoinCountHundreds) = displayThree;
                            foregroundShop(xCoinCount, yCoinCountHundredsShop) = displayThreeShop;
                        elseif moneyTotalChar(1) == '4'
                            foreground(xCoinCount, yCoinCountHundreds) = displayFour;
                            foregroundShop(xCoinCount, yCoinCountHundredsShop) = displayFourShop;
                        elseif moneyTotalChar(1) == '5'
                            foreground(xCoinCount, yCoinCountHundreds) = displayFive;
                            foregroundShop(xCoinCount, yCoinCountHundredsShop) = displayFiveShop;
                        elseif moneyTotalChar(1) == '6'
                            foreground(xCoinCount, yCoinCountHundreds) = displaySix;
                            foregroundShop(xCoinCount, yCoinCountHundredsShop) = displaySixShop;
                        elseif moneyTotalChar(1) == '7'
                            foreground(xCoinCount, yCoinCountHundreds) = displaySeven;
                            foregroundShop(xCoinCount, yCoinCountHundredsShop) = displaySevenShop;
                        elseif moneyTotalChar(1) == '8'
                            foreground(xCoinCount, yCoinCountHundreds) = displayEight;
                            foregroundShop(xCoinCount, yCoinCountHundredsShop) = displayEightShop;
                        elseif moneyTotalChar(1) == '9'
                            foreground(xCoinCount, yCoinCountHundreds) = displayNine;
                            foregroundShop(xCoinCount, yCoinCountHundredsShop) = displayNineShop;
                        elseif moneyTotalChar(1) == '0'
                            foreground(xCoinCount, yCoinCountHundreds) = displayZero;
                            foregroundShop(xCoinCount, yCoinCountHundredsShop) = displayZeroShop;
                        end
                        
                        if moneyTotalChar(2) == '1'
                            foreground(xCoinCount, yCoinCountTens) = displayOne;
                            foregroundShop(xCoinCount, yCoinCountTensShop) = displayOneShop;
                        elseif moneyTotalChar(2) == '2'
                            foreground(xCoinCount, yCoinCountTens) = displayTwo;
                            foregroundShop(xCoinCount, yCoinCountTensShop) = displayTwoShop;
                        elseif moneyTotalChar(2) == '3'
                            foreground(xCoinCount, yCoinCountTens) = displayThree;
                            foregroundShop(xCoinCount, yCoinCountTensShop) = displayThreeShop;
                        elseif moneyTotalChar(2) == '4'
                            foreground(xCoinCount, yCoinCountTens) = displayFour;
                            foregroundShop(xCoinCount, yCoinCountTensShop) = displayFourShop;
                        elseif moneyTotalChar(2) == '5'
                            foreground(xCoinCount, yCoinCountTens) = displayFive;
                            foregroundShop(xCoinCount, yCoinCountTensShop) = displayFiveShop;
                        elseif moneyTotalChar(2) == '6'
                            foreground(xCoinCount, yCoinCountTens) = displaySix;
                            foregroundShop(xCoinCount, yCoinCountTensShop) = displaySixShop;
                        elseif moneyTotalChar(2) == '7'
                            foreground(xCoinCount, yCoinCountTens) = displaySeven;
                            foregroundShop(xCoinCount, yCoinCountTensShop) = displaySevenShop;
                        elseif moneyTotalChar(2) == '8'
                            foreground(xCoinCount, yCoinCountTens) = displayEight;
                            foregroundShop(xCoinCount, yCoinCountTensShop) = displayEightShop;
                        elseif moneyTotalChar(2) == '9'
                            foreground(xCoinCount, yCoinCountTens) = displayNine;
                            foregroundShop(xCoinCount, yCoinCountTensShop) = displayNineShop;
                        elseif moneyTotalChar(2) == '0'
                            foreground(xCoinCount, yCoinCountTens) = displayZero;
                            foregroundShop(xCoinCount, yCoinCountTensShop) = displayZeroShop;
                        end
                        
                        if moneyTotalChar(3) == '1'
                            foreground(xCoinCount, yCoinCountOnes) = displayOne;
                            foregroundShop(xCoinCount, yCoinCountOnesShop) = displayOneShop;
                        elseif moneyTotalChar(3) == '2'
                            foreground(xCoinCount, yCoinCountOnes) = displayTwo;
                            foregroundShop(xCoinCount, yCoinCountOnesShop) = displayTwoShop;
                        elseif moneyTotalChar(3) == '3'
                            foreground(xCoinCount, yCoinCountOnes) = displayThree;
                            foregroundShop(xCoinCount, yCoinCountOnesShop) = displayThreeShop;
                        elseif moneyTotalChar(3) == '4'
                            foreground(xCoinCount, yCoinCountOnes) = displayFour;
                            foregroundShop(xCoinCount, yCoinCountOnesShop) = displayFourShop;
                        elseif moneyTotalChar(3) == '5'
                            foreground(xCoinCount, yCoinCountOnes) = displayFive;
                            foregroundShop(xCoinCount, yCoinCountOnesShop) = displayFiveShop;
                        elseif moneyTotalChar(3) == '6'
                            foreground(xCoinCount, yCoinCountOnes) = displaySix;
                            foregroundShop(xCoinCount, yCoinCountOnesShop) = displaySixShop;
                        elseif moneyTotalChar(3) == '7'
                            foreground(xCoinCount, yCoinCountOnes) = displaySeven;
                            foregroundShop(xCoinCount, yCoinCountOnesShop) = displaySevenShop;
                        elseif moneyTotalChar(3) == '8'
                            foreground(xCoinCount, yCoinCountOnes) = displayEight;
                            foregroundShop(xCoinCount, yCoinCountOnesShop) = displayEightShop;
                        elseif moneyTotalChar(3) == '9'
                            foreground(xCoinCount, yCoinCountOnes) = displayNine;
                            foregroundShop(xCoinCount, yCoinCountOnesShop) = displayNineShop;
                        elseif moneyTotalChar(3) == '0'
                            foreground(xCoinCount, yCoinCountOnes) = displayZero;
                            foregroundShop(xCoinCount, yCoinCountOnesShop) = displayZeroShop;
                        end
                    elseif length(moneyTotalChar) == 2
                        foreground(xCoinCount, yCoinCountHundreds) = displayZero;
                        foregroundShop(xCoinCount, yCoinCountHundredsShop) = displayZeroShop;
                        if moneyTotalChar(1) == '1'
                            foreground(xCoinCount, yCoinCountTens) = displayOne;
                            foregroundShop(xCoinCount, yCoinCountTensShop) = displayOneShop;
                        elseif moneyTotalChar(1) == '2'
                            foreground(xCoinCount, yCoinCountTens) = displayTwo;
                            foregroundShop(xCoinCount, yCoinCountTensShop) = displayTwoShop;
                        elseif moneyTotalChar(1) == '3'
                            foreground(xCoinCount, yCoinCountTens) = displayThree;
                            foregroundShop(xCoinCount, yCoinCountTensShop) = displayThreeShop;
                        elseif moneyTotalChar(1) == '4'
                            foreground(xCoinCount, yCoinCountTens) = displayFour;
                            foregroundShop(xCoinCount, yCoinCountTensShop) = displayFourShop;
                        elseif moneyTotalChar(1) == '5'
                            foreground(xCoinCount, yCoinCountTens) = displayFive;
                            foregroundShop(xCoinCount, yCoinCountTensShop) = displayFiveShop;
                        elseif moneyTotalChar(1) == '6'
                            foreground(xCoinCount, yCoinCountTens) = displaySix;
                            foregroundShop(xCoinCount, yCoinCountTensShop) = displaySixShop;
                        elseif moneyTotalChar(1) == '7'
                            foreground(xCoinCount, yCoinCountTens) = displaySeven;
                            foregroundShop(xCoinCount, yCoinCountTensShop) = displaySevenShop;
                        elseif moneyTotalChar(1) == '8'
                            foreground(xCoinCount, yCoinCountTens) = displayEight;
                            foregroundShop(xCoinCount, yCoinCountTensShop) = displayEightShop;
                        elseif moneyTotalChar(1) == '9'
                            foreground(xCoinCount, yCoinCountTens) = displayNine;
                            foregroundShop(xCoinCount, yCoinCountTensShop) = displayNineShop;
                        elseif moneyTotalChar(1) == '0'
                            foreground(xCoinCount, yCoinCountTens) = displayZero;
                            foregroundShop(xCoinCount, yCoinCountTensShop) = displayZeroShop;
                        end
                        
                        if moneyTotalChar(2) == '1'
                            foreground(xCoinCount, yCoinCountOnes) = displayOne;
                            foregroundShop(xCoinCount, yCoinCountOnesShop) = displayOneShop;
                        elseif moneyTotalChar(2) == '2'
                            foreground(xCoinCount, yCoinCountOnes) = displayTwo;
                            foregroundShop(xCoinCount, yCoinCountOnesShop) = displayTwoShop;
                        elseif moneyTotalChar(2) == '3'
                            foreground(xCoinCount, yCoinCountOnes) = displayThree;
                            foregroundShop(xCoinCount, yCoinCountOnesShop) = displayThreeShop;
                        elseif moneyTotalChar(2) == '4'
                            foreground(xCoinCount, yCoinCountOnes) = displayFour;
                            foregroundShop(xCoinCount, yCoinCountOnesShop) = displayFourShop;
                        elseif moneyTotalChar(2) == '5'
                            foreground(xCoinCount, yCoinCountOnes) = displayFive;
                            foregroundShop(xCoinCount, yCoinCountOnesShop) = displayFiveShop;
                        elseif moneyTotalChar(2) == '6'
                            foreground(xCoinCount, yCoinCountOnes) = displaySix;
                            foregroundShop(xCoinCount, yCoinCountOnesShop) = displaySixShop;
                        elseif moneyTotalChar(2) == '7'
                            foreground(xCoinCount, yCoinCountOnes) = displaySeven;
                            foregroundShop(xCoinCount, yCoinCountOnesShop) = displaySevenShop;
                        elseif moneyTotalChar(2) == '8'
                            foreground(xCoinCount, yCoinCountOnes) = displayEight;
                            foregroundShop(xCoinCount, yCoinCountOnesShop) = displayEightShop;
                        elseif moneyTotalChar(2) == '9'
                            foreground(xCoinCount, yCoinCountOnes) = displayNine;
                            foregroundShop(xCoinCount, yCoinCountOnesShop) = displayNineShop;
                        elseif moneyTotalChar(2) == '0'
                            foreground(xCoinCount, yCoinCountOnes) = displayZero;
                            foregroundShop(xCoinCount, yCoinCountOnesShop) = displayZeroShop;
                        end
                    elseif length(moneyTotalChar) == 1
                        foreground(xCoinCount, yCoinCountHundreds) = displayZero;
                        foreground(xCoinCount, yCoinCountTens) = displayZero;
                        foregroundShop(xCoinCount, yCoinCountHundredsShop) = displayZeroShop;
                        foregroundShop(xCoinCount, yCoinCountTensShop) = displayZeroShop;
                        if moneyTotalChar(1) == '1'
                            foreground(xCoinCount, yCoinCountOnes) = displayOne;
                            foregroundShop(xCoinCount, yCoinCountOnesShop) = displayOneShop;
                        elseif moneyTotalChar(1) == '2'
                            foreground(xCoinCount, yCoinCountOnes) = displayTwo;
                            foregroundShop(xCoinCount, yCoinCountOnesShop) = displayTwoShop;
                        elseif moneyTotalChar(1) == '3'
                            foreground(xCoinCount, yCoinCountOnes) = displayThree;
                            foregroundShop(xCoinCount, yCoinCountOnesShop) = displayThreeShop;
                        elseif moneyTotalChar(1) == '4'
                            foreground(xCoinCount, yCoinCountOnes) = displayFour;
                            foregroundShop(xCoinCount, yCoinCountOnesShop) = displayFourShop;
                        elseif moneyTotalChar(1) == '5'
                            foreground(xCoinCount, yCoinCountOnes) = displayFive;
                            foregroundShop(xCoinCount, yCoinCountOnesShop) = displayFiveShop;
                        elseif moneyTotalChar(1) == '6'
                            foreground(xCoinCount, yCoinCountOnes) = displaySix;
                            foregroundShop(xCoinCount, yCoinCountOnesShop) = displaySixShop;
                        elseif moneyTotalChar(1) == '7'
                            foreground(xCoinCount, yCoinCountOnes) = displaySeven;
                            foregroundShop(xCoinCount, yCoinCountOnesShop) = displaySevenShop;
                        elseif moneyTotalChar(1) == '8'
                            foreground(xCoinCount, yCoinCountOnes) = displayEight;
                            foregroundShop(xCoinCount, yCoinCountOnesShop) = displayEightShop;
                        elseif moneyTotalChar(1) == '9'
                            foreground(xCoinCount, yCoinCountOnes) = displayNine;
                            foregroundShop(xCoinCount, yCoinCountOnesShop) = displayNineShop;
                        elseif moneyTotalChar(1) == '0'
                            foreground(xCoinCount, yCoinCountOnes) = displayZero;
                            foregroundShop(xCoinCount, yCoinCountOnesShop) = displayZeroShop;
                        end
                    end

