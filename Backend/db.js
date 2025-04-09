const mongoose=require('mongoose')
require('dotenv').config()

const ATLAS_URL=process.env.ATLAS_URL
//const DB_NAME=process.env.DB_NAME

mongoose.connect(ATLAS_URL,{
   // dbName:DB_NAME
}).then(()=>{
    console.log(`Connected to MongoDB: ${ATLAS_URL}`)
}).catch((err)=>{
    console.error(`Error connecting to MongoDB: ${err}`)
})