# stuff (remember to git squash everything before publishing)

## ideas
- multiple kernel types, aka different vector norms uwu
- loss function involving zeros, not only masked values. seems better in this use case tbh
- change lgbt optimizer to the other thingy they used
- experiment with the latent space dimensionality (the thingy where you embeded da vectors)

## training fails
- initially, u and v gradients are really really small for some reason, while W and b are very out there

Parameter	Count	Mean	Std Dev	Min	Max
layers.0.layers.0.W	10	-0.744471	1.463565	-3.857394	0.007171
layers.0.layers.0.u	10	-0.005869	0.002195	-0.009772	-0.003214
layers.0.layers.0.v	10	0.413108	0.167154	0.226260	0.687869
layers.0.layers.0.b	10	-8.138438	15.442143	-39.580765	0.072118
layers.0.layers.1.W	10	-51.545993	46.368205	-146.930267	0.041123
layers.0.layers.1.u	10	-0.043063	0.074622	-0.320295	0.026075
layers.0.layers.1.v	10	0.043062	0.050827	-0.026257	0.320294
layers.0.layers.1.b	10	-88.094203	93.124628	-264.237061	0.082908
layers.1.layer.W	10	-8.646733	0.842001	-9.357000	-7.104829
layers.1.layer.u	10	-0.752334	2.297729	-2.387974	1.848223
layers.1.layer.v	10	0.010561	0.030489	-0.026256	0.033924
layers.1.layer.b	10	-16.409957	3.048358	-18.775692	-8.995026

- weights and biases are more unstable than a high school jock, while u and v are not too keen on learning - we probably have to redo the model 
- lowered learning rate - might help
- changed loss function to an unmasked loss to maximize information gain, since 0.013 percent sparsity seems bad

### first semi good training

tensor([[3.7536, 3.1596, 2.8900,  ..., 2.0011, 2.5009, 3.0001],
        [3.7536, 3.1596, 2.8900,  ..., 2.0011, 2.5009, 3.0001],
        [3.7536, 3.1596, 2.8900,  ..., 2.0011, 2.5009, 3.0001],
        ...,
        [3.7536, 3.1596, 2.8900,  ..., 2.0011, 2.5009, 3.0001],
        [3.7536, 3.1596, 2.8900,  ..., 2.0011, 2.5009, 3.0001],
        [3.7536, 3.1596, 2.8900,  ..., 2.0011, 2.5009, 3.0001]])
.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._
epoch: 29
validation rmse: tensor(0.8608) train rmse: tensor(0.8125)
validation loss:  tensor(41328.5117) , train_loss:  tensor(295799.5625)
.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._

- stopped training at epoch 8 or sth
- learned a medium representation 
- will most likely have to teach it to go over 0s as well - very sparse information

### Actually usable results no. 1 

tensor([[ 1.8103e+00,  5.0167e-01,  9.7774e-02,  ...,  1.6058e-03,
          2.1138e-03,  2.4829e-03],
        [ 3.5974e+00,  3.4830e+00,  5.5032e-01,  ..., -1.4763e-03,
          3.5846e-03, -2.7948e-03],
        [ 1.8382e+00,  1.1020e+00, -2.0108e-02,  ..., -8.9709e-04,
          1.1072e-03, -1.5930e-03],
        ...,
        [ 2.2198e+00,  2.9359e+00,  5.0100e-01,  ..., -3.2102e-04,
          7.0724e-04, -8.1535e-04],
        [ 2.2466e+00,  2.2374e+00, -2.0005e-02,  ...,  9.0622e-04,
         -1.2084e-04,  1.3102e-03],
        [ 2.4585e+00,  4.1957e-01, -1.9862e-01,  ...,  3.0739e-03,
          6.5661e-04,  4.8899e-03]])
.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._
epoch: 29
validation rmse: tensor(0.3394) train rmse: tensor(0.3133)
validation loss:  tensor(745368.3750) , train_loss:  tensor(3205944.2500)
.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._

- changed loss function to go over 0s as well as actual grades - actually surprisingly worked
- did not learn a mean representation of columns
- could use some hyperpameter optimization
- will experiment with weighted loss scaled by sparsity for 0s in the thingy

## żabson - sexoholik lyrics
Swizzy na bicie ziomal
Sexoholik!
Grzeczne dupy nie chcą ze mną chodzić
Niegrzeczne dupy chcą się pierdolić
Ona chce uprawiać seks, nie po to żeby dziecko zrobić
Ona chce uprawiać seks, by być jak te z pornoli
Trzecia w nocy jest, mój telefon dzwoni
To jedna z moich ex albo jedna z Twoich
To jedna z moich ex nażarła się molly
Chce uprawiać ze mną seks w wielkiej metropolii
By wyrywać dupy z klubu, nie opuszczam czterech ścian
Ona gdy wychodzi stamtąd, najebana dzwoni sama
Na, na, na, na, na, na, na, na, na, robi spam
Dobrze wiem, co od niej dziś dostanę, ona dobrze wie, co dla niej mam
Holik! Nie dzwoń do mnie, jeśli nie wiesz o co chodzi
Środek nocy to nie pora na rozmowy
Co chcesz zrobić? Co chcesz zbroić?
Chcesz się kłócić, czy chcesz się godzić?
Ja nie chcę się dochodzić, chcę dochodzić
Seks dla przyjemności, ale nie myśl, że zarobisz
Dupy dają te eskortki, lecz nie jako klientowi, o
Zdejmuję jej majtki od Victorii
W znak Victorii rozkładam jej nogi
Sexoholik!
Grzeczne dupy nie chcą ze mną chodzić
Niegrzeczne dupy chcą się pierdolić
Ona chce uprawiać seks, nie po to żeby dziecko zrobić
Ona chce uprawiać seks, by być jak te z pornoli
Trzecia w nocy jest, mój telefon dzwoni
To jedna z moich ex albo jedna z Twoich
To jedna z moich ex nażarła się molly
Chce uprawiać ze mną seks w wielkiej metropolii
Zawsze bezpieczny seks, wiesz jak jest, wchodzę tylko w gumkach
Ona prosi, żebym wszedł w nią bez, po dwóch pocałunkach
Suko! Weź ze mnie zejdź, no bo jesteś brudna
Bleh, bleh, bleh, bleh, bleh, nie chcę złapać gówna
Nie rozdrabniam się, jakbym kruszył topka
Ale sama dobrze wiesz, że masz być schludna i pachnąca (fresh)
Każda moja suka specjalnie dla mnie wygląda
Lubię białe, lubię małe, lubię duży rozmiar
Lubię czekoladę tak jak Willy Wonka
One są tu przekonane, co je we mnie w noc pociąga
Bo czuje się jak boss, mówię im wprost
Nie biorę kredytów na procent, bo pluję na sos
Dlatego noszę na sobie te Dolce, dlatego noszę Dior
Bo jestem najbardziej stylowym fuckerem w Polsce, one to widzą
Fucker! Nie raper
Sexoholik!
## We don luv em - Hoodrich Pablo Juan lyrics
Ooh, yeah
MONY POWR RSPT, nigga
It's a money set, you know what I'm saying?
Everybody getting money, nigga
Yeah, Pablo Juan
The money go where I go
Smoking on gelato
Foreign car swerving potholes
Bad bitch, she from Chicago
She freaky, she gon' bust it
She thick as fuck, I'm lusting
I got her from my cousin
So what? 'Cause we don't love 'em
Fuck that, I wanna hit from the back
Backwoods smoking, it's fat
Dressing like I got a sack
I pull up, jumped out the bach
Bad bitch and her ass fat
Four-door Coupe, it got a hatch
On the Xans, I might crash that (yuh)
Car got gadgets, my bitches got asses
Expensive glasses like I'm teaching class
Too fresh to take out the trash
Fresh to death, where is my casket?
I always stay with assassins
I'm always late with the fashion
Teacher gave me an F, that's fantastic
VS diamonds on me, look how they flashing
Rocking Saint Laurent, I guess I be dabbing
I got the Louis V, Supreme collabbing
Bought a mansion way away like a cabin
Taking off my swag, I feel like your daddy
You a beggar, I'm a hustler
I'm the dealer, you the customer
Ketchup, little nigga, I'm mustard
Smoking the Backwoods, they coming from Russia
I ain't never really trust you
Knew I should've never trust you (hell nah)
You ain't real, you a busta
These niggas was always sus
These niggas start snitching for nothing
These niggas wanna live by the gun
Guess what? You gon' get what you want
El Patrón, nigga, I want a ton (yuh)
The money go where I go
Smoking on gelato
Foreign car swerving potholes
Bad bitch, she from Chicago
She freaky, she gon' bust it
She thick as fuck, I'm lusting
I got her from my cousin
So what? 'Cause we don't love 'em
Fuck that, I wanna hit from the back
Backwoods smoking, it's fat
Dressing like I got a sack (ooh)
I pull up, jumped out the bach
Bad bitch and her ass fat
Four-door Coupe, it got a hatch
On the Xans, I might crash that
Pull up on you, just send me the Addy
Bad bitch call me daddy
Xan, Perc, and a Addy
I really wanna fuck a Kardashian
I like a freaky bitch that's gon' suck it
I just be kicking shit like it was rugby
Hell no, baby, don't call me hubby
Fuck you thought, baby? We was just fucking
Ooh, I'm back to the trap and I'm serving that
I done got me a sack like a running back
Two pints of Hi-Tech and a eighth of Act
I'ma fuck on your bitch, I'ma break her back
I'ma fuck on your bitch, I'ma give her back
I got two bitches playing Pitty Pat
I just do it like the Nike check
My neck froze, got a ice attack
The money go where I go
Smoking on gelato
Foreign car swerving potholes
Bad bitch, she from Chicago
She freaky, she gon' bust it
She thick as fuck, I'm lusting
I got her from my cousin
So what? 'Cause we don't love 'em
Fuck that, I wanna hit from the back
Backwoods smoking, it's fat
Dressing like I got a sack (ooh)
I pull up, jumped out the bach
Bad bitch and her ass fat
Four-door Coupe, it got a hatch
On the Xans, I might crash that
## Żabson - trapczan lyrics
O czym ty mi kurwa gadasz? (Jarasz trawę w NOBOCOTO)
Co ty Yah00 odpierdalasz? (Jarasz trawę, chociaż to jest zakazane)
Garaż (chociaż to jest...)
Rzucam tą sukę na tapczan (woo)
Robię jej na dupie touchdown (woah, woah)
Kiedy ty wchodzisz na 4chan (wow, wow)
Wtedy ja zamawiam jej Bolta (waa, prr)
Robię wsad - Marcin Gortat (slatt, slatt)
Palę stuff, ona pyta, "Dasz mi jointa?"
Mała dawka ją zmiotła, moja trawka jest za mocna
Gdy wchodzę se do klubu, biorę panie
Gdy jadę se na koncert robię pranie
To nie biały proszek, to nie biały nosek
To nie biały - i tak jestem pojebany
Te dupy oniemiały, kiedy wszedłem
Będą krzyczały, kiedy wejdę
Same się pchały, na te backstage
Każda z nich chce być moją bejbe
Robię wszystko, żeby mieć codziennie payday
Robię wszystko niezależnie, jebać label
Ona kręci swoją dupą tak jak Beyblade
I chce zobaczyć z bardzo bliska moje meble
Rzucam tą sukę na tapczan (woo)
Robię jej na dupie touchdown (woah, woah)
Kiedy ty wchodzisz na 4chan (wow, wow)
Wtedy ja zamawiam jej Bolta (waa, prr)
Robię wsad - Marcin Gortat (slatt, slatt)
Palę stuff, ona pyta, "Dasz mi jointa?"
Mała dawka ją zmiotła, moja trawka jest za mocna
Biorę ją na traphouse, rzucam ją na trapczan
Rach ciach, ciach, ciach, szybka akcja
Ona nie chce z Tobą wsiąść do Twojego auta
Ona nie chce z Tobą wsiąść, bo Volkswagen Passat
Zamawiam Ubera Blacka, na przodzie gwiazda
Ona chce obok mnie siąść, chociaż nie mam prawka
To była akcja łatwa, z mleczkiem kaszka
Ona rozkłada szybko nogi, jak z Ikei szafka
Puf puf - strzelam Buckshot
Puf puf - na meble Agatka
Rzucam tą sukę na tapczan
Robię jej na dupie touchdown
Kiedy ty wchodzisz na 4chan (wow, wow)
Wtedy ja zamawiam jej Bolta (waa, prr)
Robię wsad - Marcin Gortat (slatt, slatt)
Palę stuff, ona pyta, "Dasz mi jointa?"
Mała dawka ją zmiotła, moja trawka jest za mocna
