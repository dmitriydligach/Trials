# cTAKES preprocessing

git clone https://github.com/dmitriydligach/ctakes-misc.git

cd ctakes-misc/
mvn clean compile

mvn exec:java -Dexec.mainClass="org.apache.ctakes.pipelines.UmlsLookupPipeline" -Dexec.args="--input-dir /Users/Dima/Loyola/Data/Trials/Test/Text/ --output-dir /Users/Dima/Loyola/Data/Trials/Test/Xmi/" -Dctakes.umlsuser=<user name> -Dctakes.umlspw=<pwd>

mvn exec:java -Dexec.mainClass="org.apache.ctakes.consumers.ExtractCuis" -Dexec.args="--xmi-dir /Users/Dima/Loyola/Data/Trials/Test/Xmi/ --output-dir /Users/Dima/Loyola/Data/Trials/Test/Cuis/" -Dctakes.umlsuser=<user name> -Dctakes.umlspw=<pwd>
