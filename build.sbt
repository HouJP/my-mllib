name := "my-mllib"

version := "1.0"

scalaVersion := "2.10.4"

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.5.2"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.5.2"
libraryDependencies += "com.github.scopt" %% "scopt" % "3.3.0"
libraryDependencies += "org.scalatest" %% "scalatest" % "2.2.1" % "test"


resolvers += Resolver.sonatypeRepo("public")
