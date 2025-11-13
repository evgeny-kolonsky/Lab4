Measuring p-Ge samples heat load

Preocedure:
1. Without B-load seacrh Ip where Up = 0 as close as possible( ~1-5 mV).
2. Having Up = 0 rotate UH_COMP to set Uh = 0 as small as possibe (< 1mV)
3. Make zero masuerements: save points from Ip = -30 mA tp Ip = 30 mA. SAve to file `zeroXX.txt`
4. Hot load at Ip = 30 mA:
   4.1 Set B = -250 mT. Remove probe
   4.2 Run Temperature load load. Sve points both for heating and cooling. Measure and fix B at final point
   4.3 Change polarity. B might me have shifted upwards, up to 280 mT. Set B = 250 mT again
   4.4 Run and save temperature load, both for heating and cooling
5. Analyze data
