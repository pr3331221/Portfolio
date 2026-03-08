import lang.stride.*;
import greenfoot.*;
import java.util.*;
/**
* Write a description of class FlappyCheesy here.
* @author (your name) @version (a version number or a date)
*/
public class FlappyCheesy extends Actor
{
/* (World, Actor, GreenfootImage, Greenfoot and MouseInfo)*/
protected double dy = 0;
protected double i = 1.3;
protected double j = -15;
protected int counter = 0;
/**
* Act - do whatever the FlappyWass wants to do. This method is called whenever the 'Act' or
'Run' button gets pressed in the environment.
*/
public FlappyCheesy()
{
GreenfootImage doake = getImage();
doake.scale((int)(doake.getWidth() * 3), (int)(doake.getHeight() * 3));
setImage(doake);
}
/* public void addScore(){ counter++; }*/
/**
*
*/
public void act()
{
if (getWorld() instanceof background) {
setLocation(getX(), (int)(getY() + dy));
if (Greenfoot.isKeyDown("space") == true) {
dy = j;
Greenfoot.playSound("sfx_wing.wav");
}
if (isTouching(Coin.class)) {
World myWorld = getWorld();

background back = (background)myWorld;
Score score = back.getScore();
score.setScore();
removeTouching(Coin.class);
Greenfoot.playSound("sfx_point.wav");
}
}
/* List list = new ArrayList(); list = getObjectsInRange(90, TopPost.class);
if(!list.isEmpty()){*/
if (getOneIntersectingObject(TopPost.class) != null) {
GameOver gameover = new GameOver();
getWorld().addObject(gameover, getWorld().getWidth() / 2, getWorld().getHeight() / 2);
Greenfoot.playSound("sfx_hit.wav");
Greenfoot.playSound("sfx_die.wav");
Greenfoot.stop();
}
/* List list2 = new ArrayList(); list2 = getObjectsInRange(90, BottomPost.class);
if(!list2.isEmpty()){*/
if (getOneIntersectingObject(BottomPost.class) != null) {
GameOver gameover = new GameOver();
getWorld().addObject(gameover, getWorld().getWidth() / 2, getWorld().getHeight() / 2);
Greenfoot.playSound("sfx_hit.wav");
Greenfoot.playSound("sfx_die.wav");
Greenfoot.stop();
}
if (getWorld() instanceof background) {
if (dy > -10 && dy < 10) {
setRotation(30);
}
else if (dy < -10) {
setRotation(-30);
}
else if (dy > 10) {
setRotation(30);
}
}
if (getY() > getWorld().getHeight() + 50 || getY() < - (getWorld().getHeight()) - 50) {
GameOver gameover = new GameOver();
getWorld().addObject(gameover, getWorld().getWidth() / 2, getWorld().getHeight() / 2);
Greenfoot.playSound("sfx_hit.wav");
Greenfoot.playSound("sfx_die.wav");
Greenfoot.stop();
}

dy = dy + i;
}
}
import greenfoot.*; // (World, Actor, GreenfootImage, Greenfoot and MouseInfo)
/**
* Write a description of class Coin here.
*
* @author (your name)
* @version (a version number or a date)
*/
public class Coin extends Actor
{
int dx = -4;
/**
* Act - do whatever the Coin wants to do. This method is called whenever
* the 'Act' or 'Run' button gets pressed in the environment.
*/
public Coin(){
GreenfootImage coin = getImage();
coin.scale(coin.getWidth()/10, coin.getHeight()/10);
setImage(coin);
}
public void act()
{
setLocation(getX() + dx, getY());
}
}
import greenfoot.*; // (World, Actor, GreenfootImage, Greenfoot and MouseInfo)
/**
* Write a description of class BottomPost here.
*
* @author (your name)
* @version (a version number or a date)
*/
public class BottomPost extends Actor
{
int dx = -4;

/**
* Act - do whatever the BottomPost wants to do. This method is called whenever
* the 'Act' or 'Run' button gets pressed in the environment.
*/
public BottomPost(){
GreenfootImage bottom = getImage();
bottom.scale(bottom.getWidth()*2, bottom.getHeight()*2);
setImage(bottom);
}
public void act()
{
setLocation(getX() + dx, getY());
}
}
import greenfoot.*; // (World, Actor, GreenfootImage, Greenfoot and MouseInfo)
/**
* Write a description of class StartRoom here.
*
* @author (your name)
* @version (a version number or a date)
*/
public class StartRoom extends World
{
private FlappyCheesy flappy;
private Start start;
/**
* Constructor for objects of class StartRoom.
*
*/
public StartRoom()
{
// Create a new world with 600x400 cells with a cell size of 1x1 pixels.
super(1200, 800, 1);
prepare();
}
public void act(){
if(Greenfoot.isKeyDown("x")){
Greenfoot.setWorld(new background());
}
}
public void prepare(){

GreenfootImage startImage = new GreenfootImage("flap.jpg");
startImage.scale(getWidth(),getHeight());
setBackground(startImage);

TItle title = new TItle();
addObject(title, getWidth()/2, (getHeight()/2)-200);
pressX pressXobject = new pressX();
addObject(pressXobject, (getWidth()/2) -50,((getHeight()/2)+100));
start = new Start();
addObject(start, (getWidth()/2)+50 ,(getHeight()/2) +100);
flappy = new FlappyCheesy();
addObject(flappy, 600,400);
}
}
import greenfoot.*; // (World, Actor, GreenfootImage, Greenfoot and MouseInfo)
/**
* Write a description of class background here.
*
* @author (your name)
* @version (a version number or a date)
*/
public class background extends World
{
private int count = 0;
private int counter = 0;
Score score = new Score();
public Score getScore(){
return score;
}
/**
* Constructor for objects of class background.
*
*/
public background(){
// Create a new world with 600x400 cells with a cell size of 1x1 pixels.

super(1200, 800, 1, false);
setPaintOrder(GameOver.class, FlappyCheesy.class, Coin.class,TopPost.class,
BottomPost.class, Score.class);
FlappyCheesy flappy= new FlappyCheesy();
addObject(flappy, 100, getHeight()/2);

addObject(score, 200, 100);

addObjects();
}
public void act(){
count++;
if(count==150){
int randomNumber = (int)(Math.random()*(801));
TopPost top = new TopPost();
BottomPost bottom = new BottomPost();
GreenfootImage bottomImage = bottom.getImage();
GreenfootImage topImage = top.getImage();
addObject(top, getWidth(), (randomNumber / 4)-75);
addObject(bottom, getWidth(), getHeight()+randomNumber/4);
count = 0;
}
if(count==75){
int randomNumber = (int)(Math.random()*(775-25)+1)+25;
Coin x = new Coin();
GreenfootImage coinImage = x.getImage();
addObject(x, getWidth(), randomNumber);
}
}
public void addObjects() {
GreenfootImage bg = new
GreenfootImage("a-kitchen-restaurant-background_bec26d6b-ac9d-412b-8fed-7fe8bdecc78e_g
rande.jpg");
bg.scale(getWidth(), getHeight());
setBackground(bg);
}
}

import greenfoot.*; // (World, Actor, GreenfootImage, Greenfoot and MouseInfo)
/**
* Write a description of class TopPost here.
*
* @author (your name)
* @version (a version number or a date)
*/
public class TopPost extends Actor
{
int dx = -4;
/**
* Act - do whatever the TopPost wants to do. This method is called whenever
* the 'Act' or 'Run' button gets pressed in the environment.
*/
public void act()
{
setLocation(getX() + dx, getY());
}
public TopPost(){
GreenfootImage top = getImage();
top.scale(top.getWidth()*2, top.getHeight()*2);
setImage(top);
}
}
import greenfoot.*; // (World, Actor, GreenfootImage, Greenfoot and MouseInfo)
/**
* Write a description of class Score here.
*
* @author (your name)
* @version (a version number or a date)
*/
public class Score extends Actor
{
/**
* Act - do whatever the Score wants to do. This method is called whenever
* the 'Act' or 'Run' button gets pressed in the environment.
*/

int score = 0;
public void updateImage()
{
setImage(new GreenfootImage("Score: " + score, 50, Color.WHITE, new Color(0,0,0,0)));
}
public void setScore(){
score++;
updateImage();
}
public Score(){
updateImage();
}
public void act()
{
}
}
import greenfoot.*; // (World, Actor, GreenfootImage, Greenfoot and MouseInfo)
/**
* Write a description of class GameOver here.
*
* @author (your name)
* @version (a version number or a date)
*/
public class GameOver extends Actor
{
/**
* Act - do whatever the GameOver wants to do. This method is called whenever
* the 'Act' or 'Run' button gets pressed in the environment.
*/
public void act()
{
// Add your action code here.
}
}
import greenfoot.*; // (World, Actor, GreenfootImage, Greenfoot and MouseInfo)
/**
* Write a description of class Start here.

*
* @author (your name)
* @version (a version number or a date)
*/
public class Start extends Actor
{
/**
* Act - do whatever the Start wants to do. This method is called whenever
* the 'Act' or 'Run' button gets pressed in the environment.
*/
public void act()
{
// Add your action code here.
}
}
import greenfoot.*; // (World, Actor, GreenfootImage, Greenfoot and MouseInfo)
/**
* Write a description of class TItle here.
*
* @author (your name)
* @version (a version number or a date)
*/
public class TItle extends Actor
{
/**
* Act - do whatever the TItle wants to do. This method is called whenever
* the 'Act' or 'Run' button gets pressed in the environment.
*/
public void act()
{
}
public TItle(){
GreenfootImage title = getImage();
title.scale(title.getWidth()*5, title.getHeight()*5);
setImage(title);
}
}
import greenfoot.*; // (World, Actor, GreenfootImage, Greenfoot and MouseInfo)

/**
* Write a description of class pressX here.
*
* @author (your name)
* @version (a version number or a date)
*/
public class pressX extends Actor
{
/**
* Act - do whatever the pressX wants to do. This method is called whenever
* the 'Act' or 'Run' button gets pressed in the environment.
*/
public void act()
{
//Add your action code here.
}
}
